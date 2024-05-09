

import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import logging
from torch.nn import init
from model.BMMTNet import BmmtNetV7
import numpy as np

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

model = init_net(BmmtNetV7(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2, 3]),gpu_ids=[])

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