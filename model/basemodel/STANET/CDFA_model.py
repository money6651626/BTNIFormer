import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel.STANET.backbone import define_F,CDSA


class CDFAModel(nn.Module):
    """
    change detection module:
    feature extractor+ spatial-temporal-self-attention
    contrastive loss
    """

    def __init__(self,embed_dim=64,arch="mynet3",ds=1,SA_mode='BAM'):
        super(CDFAModel, self).__init__()
        self.ds = 1
        self.netF = define_F(in_c=3, f_c=embed_dim, type=arch)
        self.netA = CDSA(in_c=embed_dim, ds=ds, mode=SA_mode)


    def forward(self,x1,x2):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        feat_A = self.netF(x1)  # f(A)
        feat_B = self.netF(x2)   # f(B)
        feat_A, feat_B = self.netA(feat_A,feat_B) #STAttention
        feat_A = feat_A.permute(0, 2, 3, 1)
        feat_B = feat_B.permute(0, 2, 3, 1)
        dist = F.pairwise_distance(feat_A, feat_B, keepdim=True).permute(0, 3, 1, 2)  # 特征距离
        dist = F.interpolate(dist, size=x1.shape[2:], mode='bilinear',align_corners=True)
        return dist



if __name__ == "__main__":

    model = CDFAModel().cuda()

    dummy_input = torch.randn(1, 3, 256, 256).cuda()


    out=model(dummy_input,dummy_input)
    print(out.shape)