import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from .decoder import build_decoder
from .utils import CBAM, DS_layer
import logging
import os
logger = logging.getLogger(__name__)

class DSAMNet(nn.Module):
    def __init__(self, n_class=2,  ratio = 8, kernel = 7, backbone='resnet18', output_stride=16, f_c=64, freeze_bn=False, in_c=3):
        super(DSAMNet, self).__init__()
        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, in_c) #build resnet18
        self.decoder = build_decoder(f_c, BatchNorm)

        self.cbam0 = CBAM(f_c, ratio, kernel)
        self.cbam1 = CBAM(f_c, ratio, kernel)

        self.ds_lyr2 = DS_layer(64, 32, 2, 1, n_class)
        self.ds_lyr3 = DS_layer(128, 32, 4, 3, n_class)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x1, x2): #input batch×3×256×256
        x_1, f2_1, f3_1, f4_1 = self.backbone(x1) #512×32×32;64×128×128;128×64×64;256×32×32
        x_2, f2_2, f3_2, f4_2 = self.backbone(x2)

        x1 = self.decoder(x_1, f2_1, f3_1, f4_1)#64×128×128
        x2 = self.decoder(x_2, f2_2, f3_2, f4_2)

        x1 = self.cbam0(x1) #64×128×128
        x2 = self.cbam1(x2) # channel = 64

        #1.11
        x1=x1.transpose(1,-1)
        x2=x2.transpose(1,-1)

        dist = F.pairwise_distance(x1, x2, keepdim=True).permute(0,-1,1,2)# channel = 1  64×128×1
        out=x1.shape[1:3]
        dist = F.interpolate(dist, size=(256,256), mode='bilinear', align_corners=True) #64×256×256

        ds2 = self.ds_lyr2(torch.abs(f2_1 - f2_2))
        ds3 = self.ds_lyr3(torch.abs(f3_1 - f3_2))

        return dist, ds2, ds3


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def init_weights(self, pretrained_path='', ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained_path):
            pretrained_dict = torch.load(pretrained_path)
            logger.info('=> loading pretrained model {}'.format(pretrained_path))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)
            return (missing_keys, unexpected_keys)