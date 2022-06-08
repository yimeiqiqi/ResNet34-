from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import json
from datetime import timedelta
from ResNet import resnet34
# read class_indict
try:
    json_file = open('../flower_data/class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = resnet34(num_classes=5)
# load model weights
model_path = '../flower_data/ResNet34_cbam.pth'
# , map_location='cpu'
model.load_state_dict(torch.load(model_path, map_location='cpu'))


# 关闭 Dropout
model.eval()


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

# 获取文件名后缀
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


# 图片装换操作
def tran(img_path):
    # 预处理
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # load image
    img = Image.open(img_path).convert('RGB')
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    return img


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    # file_name = ""
    if request.method == 'POST':
        file = request.files['file']
        if not (file and allowed_file(file.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
#         将上传的图片保存到页面中展示
        with open(r'./static/images/val.png', 'wb') as val:
            val.write(file.read())
        file.close()
        base_path = os.path.dirname(__file__)  # 当前文件所在路径
        # file_name = secure_filename(file.filename)  # 文件名称
        upload_path = os.path.join(base_path, 'static/images/val.png')  # 文件完整路径
        img = tran(upload_path)
        # 预测图片
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img))  # 将输出压缩，即压缩掉 batch 这个维度
            predict = torch.softmax(output, dim=0)
            print(predict)
            predict_cla = torch.argmax(predict).numpy()
            res_chinese = ["小雏菊", "蒲公英", "玫瑰花", "向日葵", "郁金香"]
            max_name = res_chinese[predict_cla]  # 最可能的花
            max_acc = round(predict[predict_cla].item()*100, 2)  # 保留两位小数
#             准确率大于90%，只输出一种结果
            if max_acc > 90:
                return render_template('result.html', max_name=max_name, max_acc=max_acc)
            else:
                predict[predict_cla] = 0  # 最大值归零 以便求第二大
                sec_cla = 0
                for i in range(1, len(predict)):
                    if predict[sec_cla].item() < predict[i].item():
                        sec_cla = i
                sec_name = res_chinese[sec_cla]
                sec_acc = round(predict[sec_cla].item()*100, 2)
#                 不足百分之九十，输出两种准确率最大的结果
                return render_template('result2.html', max_name=max_name, max_acc=max_acc,
                                       sec_acc=sec_acc, sec_name=sec_name)
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
