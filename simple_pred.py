

import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import logging
from model.BMMTNet import BmmtNetV7
import numpy as np

model = BmmtNetV7(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2, 3])
checkpoint = torch.load("pretrain_weight/WHU_CD.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
#model = torchvision.models.resnet18(pretrained=True)  # 数量对不上




UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif','tif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def MyPredict(image_path_1,image_path_2):

    # Your data transformation
    normal_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    model.eval()
    # Load and transform the single image

    image1 = Image.open(image_path_1).convert('RGB')
    input_image1 = normal_transforms(image1).unsqueeze(0)
    image2= Image.open(image_path_2).convert('RGB')
    input_image2 = normal_transforms(image2).unsqueeze(0)
    # Make prediction
    with torch.no_grad():
        output = model(input_image1,input_image2)

    # Get the predicted label
    pred_out = torch.argmax(output,dim=1)

    return pred_out*255



def upload_file():

    result = MyPredict(os.path.join(UPLOAD_FOLDER,"A","whucd_00698.png"),os.path.join(UPLOAD_FOLDER,"B","whucd_00698.png"))
    result_np=result.permute(1, 2, 0).numpy().astype(np.uint8)

    result_path=os.path.join(UPLOAD_FOLDER,"out","whucd_00698.png")
    im = Image.fromarray(np.squeeze(result_np))#灰度图是（h,w）
    im.save(result_path)






if __name__ == "__main__":
    upload_file()