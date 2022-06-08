import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import os
import json
import time
from tensorboardX import SummaryWriter
from torchvision import models
from ResNet import resnet34

writer = SummaryWriter('../../tf-logs/')  # tensorboard展示数据
# 使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪，再缩放成 224×224
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),

    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}

# 获取图像数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../flower_data"))  # get data root path 返回上上层目录
image_path = data_root + "/flower_data/"  # flower data_set path

# 导入训练集并进行预处理
train_dataset = datasets.ImageFolder(root=image_path + "train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,  # 导入的训练集
                                           batch_size=16,  # 每批训练的样本数
                                           shuffle=True,  # 是否打乱训练集
                                           num_workers=4,  # 使用线程数
                                           pin_memory=True)
# 导入、加载 验证集
# 导入验证集并进行预处理
validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,  # 导入的验证集
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=4,  # 使用线程数
                                              pin_memory=True)

# 存储 索引：标签 的字典
# 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# 将 flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())
# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
with open('../flower_data/class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 加入 迁移学习
model_resnet = models.resnet34(pretrained=True)
pretrained_dict = model_resnet.state_dict()
train_resnet = resnet34(num_classes=5)
train_dict = train_resnet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in train_dict}  # 删除网络层不一样的
train_dict.update(pretrained_dict)  # 更新成修改后的网络
train_resnet.load_state_dict(train_dict)

train_resnet.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()  # 交叉熵损失
loss_function.to(device)
optimizer = optim.Adam(train_resnet.parameters(), lr=0.0001)  # 优化器（训练参数，学习率, 权重缩减）

save_path = '../flower_data/ResNet34_cba.pth'
best_val_acc = 0.0
best_tra_acc = 0.0
for epoch in range(500):
    train_resnet.train()  # 训练过程中开启 Dropout
    tra_sum_loss = 0.0  # 每个 epoch 都会对 running_loss  清零
    val_sum_loss = 0.0
    time_start = time.perf_counter()  # 对训练一个 epoch 计时
    tra_sum_acc = 0.0

    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        images, tra_labels = data  # 获取训练集的图像和标签
        optimizer.zero_grad()  # 清除历史梯度
        tra_outputs = train_resnet(images.to(device))  # 正向传播
        tra_loss = loss_function(tra_outputs, tra_labels.to(device))  # 计算损失
        tra_loss.backward()  # 反向传播
        optimizer.step()  # 优化器更新参数
        tra_predict = torch.max(tra_outputs, dim=1)[1].to(device)  # 以output中值最大位置对应的索引（标签）作为预测输出
        tra_sum_acc += (tra_predict == tra_labels.to(device)).sum().item()
        tra_sum_loss += tra_loss.item()
    tra_acc = tra_sum_acc / train_num
    if tra_acc > best_tra_acc:
        best_tra_acc = tra_acc
    writer.add_scalar("best/train", best_tra_acc, epoch)
    writer.add_scalar('loss/train', tra_sum_loss, epoch)
    print('单次时间：%f s' % (time.perf_counter() - time_start))
    train_resnet.eval()  # 验证过程中关闭 Dropout
    val_sum_acc = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            val_outputs = train_resnet(val_images.to(device))
            val_loss = loss_function(val_outputs, val_labels.to(device))
            val_predict = torch.max(val_outputs, dim=1)[1].to(device)  # 以output中值最大位置对应的索引（标签）作为预测输出
            val_sum_acc += (val_predict == val_labels.to(device)).sum().item()
            val_sum_loss += val_loss.item()
        val_acc = val_sum_acc / val_num
        writer.add_scalar("loss/val", val_sum_loss, epoch)
        # 保存准确率最高的那次网络参数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(train_resnet.state_dict(), save_path)
        with open(os.path.join("../flower_data/ResNet34_train.log"), "a") as log:
            log.write(str('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n'
                          % (epoch + 1, tra_sum_loss / step, val_acc)) + "\n")
        writer.add_scalar("best/val", best_val_acc, epoch)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n'
              % (epoch + 1, tra_sum_loss / step, val_acc))
    writer.close()
with open(os.path.join("../flower_data/ResNet.log"), "a") as log:
    log.write(str('Finished Training.time:%f s') + " % (time.perf_counter())\n")
print('Finished Training')
