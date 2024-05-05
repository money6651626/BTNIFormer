import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import functools
from einops import rearrange

from model.basemodel import resnet

from model.basemodel.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d
from model.basemodel.ChangeFormer import ChangeFormerV6,ChangeFormerV6_b1,ChangeFormerV6_b2
from model.basemodel.SiamUnet_diff import SiamUnet_diff
from model.basemodel.SiamUnet_conc import SiamUnet_conc
from model.basemodel.Unet import Unet
from model.basemodel.DTCDSCN import CDNet34
from model.basemodel.DSAMNEet.dsamnet import DSAMNet
from model.basemodel.STANET.CDFA_model import CDFAModel
from model.basemodel.Tiny_CD.models.change_classifier import  ChangeClassifier
from model.BMMTNet import BmmtNetV1, BmmtNetV2, BmmtNetV3, BmmtNetV5, BmmtNetV6, BmmtNetV7,BmmtNet_Res34,BmmtNet_segformer_b2,BmmtNet_segformer_b1


###############################################################################
# Helper Functions
###############################################################################

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs // 3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


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


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'base_resnet18':
        net = ResNet(input_nc=3, output_nc=2, output_sigmoid=False)

    elif args.net_G == 'base_transformer_pos_s4':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned')

    elif args.net_G == 'base_transformer_pos_s4_dd8':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8)

    elif args.net_G == 'Bit':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                               with_pos='learned', enc_depth=1, dec_depth=8,
                               decoder_dim_head=8)  # base_transformer_pos_s4_dd8_dedim8


    elif args.net_G == 'Tiny_CD':
        net = ChangeClassifier()

    elif args.net_G == 'ChangeFormerV6':
        net = ChangeFormerV6(
            embed_dim=256)  # ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    elif args.net_G == 'ChangeFormerV6_b1':
        net = ChangeFormerV6_b1(
            embed_dim=64)  # ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    elif args.net_G == 'ChangeFormerV6_b2':
        net = ChangeFormerV6_b2(
            embed_dim=64)  # ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)

    elif args.net_G == "SiamUnet_diff":
        # Implementation of ``Fully convolutional siamese networks for change detection''
        # Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = SiamUnet_diff(input_nbr=3, label_nbr=2)

    elif args.net_G == "SiamUnet_conc":
        # Implementation of ``Fully convolutional siamese networks for change detection''
        # Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = SiamUnet_conc(input_nbr=3, label_nbr=2)

    elif args.net_G == "Unet":
        # Usually abbreviated as FC-EF = Image Level Concatenation
        # Implementation of ``Fully convolutional siamese networks for change detection''
        # Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
        net = Unet(input_nbr=3, label_nbr=2)

    elif args.net_G == "DTCDSCN":
        # The implementation of the paper"Building Change Detection for Remote Sensing Images Using a Dual Task Constrained Deep Siamese Convolutional Network Model "
        # Code copied from: https://github.com/fitzpchao/DTCDSCN
        net = CDNet34(in_channels=3)

    elif args.net_G == "STANet":
        net = CDFAModel(embed_dim=64)

    elif args.net_G == "DSAMNet_34":
        net = DSAMNet(n_class=2, backbone="resnet34")
    elif args.net_G == "BmmtNet_NAT_V1":
        net = BmmtNetV1()  # base版本，不使用exchange，不适用cross_attention 验证DiNAT比Segformer好
    elif args.net_G == "BmmtNet_NAT_V2":
        net = BmmtNetV2(exchange_type="and", exchange_layer=[0, 1, 2, 3],
                        freature_cross_layer=[])  # 使用四个尺度的空间交换 验证空间交换的优势
    elif args.net_G == "BmmtNet_NAT_V2-1":
        net = BmmtNetV2(exchange_type="and", exchange_layer=[1, 2, 3], freature_cross_layer=[])  # 使用四个尺度的空间交换 验证空间交换的优势
    elif args.net_G == "BmmtNet_NAT_V2-2":
        net = BmmtNetV2(exchange_type="single", exchange_layer=[0, 1, 2, 3],
                        freature_cross_layer=[])  # 使用四个尺度的空间交换 验证空间交换的优势
    elif args.net_G == "BmmtNet_NAT_V4":
        net = BmmtNetV2(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2, 3])
        # 四个尺度的crossattention（只涉及多尺度特征的输出交互类似于SIAMIXFORMER） 验证crossattention的优势 （如果exchange有效就加上）
    elif args.net_G == "BmmtNet_NAT_V4-1":
        net = BmmtNetV2(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[1, 2, 3])
        # 四个尺度的crossattention（只涉及多尺度特征的输出交互类似于SIAMIXFORMER） 验证crossattention的优势 （如果exchange有效就加上）
    elif args.net_G == "BmmtNet_NAT_V3":
        net = BmmtNetV3(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2])
        # cross attention介入特征提取的过程 验证changer方式交互的优势
    elif args.net_G == "BmmtNet_NAT_V3+exhange":
        net = BmmtNetV3(exchange_type="and", exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2])
        # cross attention介入特征提取的过程 验证changer方式交互的优势
    elif args.net_G == "BmmtNet_NAT_V3-1":
        net = BmmtNetV3(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[1, 2])
        # cross attention介入特征提取的过程 验证changer方式交互的优势
    elif args.net_G == "BmmtNet_NAT_V5":
        net = BmmtNetV5()
    elif args.net_G == "BmmtNet_NAT_V6":
        net = BmmtNetV6()
    elif args.net_G == "BmmtNet_NAT_V7":
        net = BmmtNetV7(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2, 3])
    elif args.net_G == "BmmtNet_NAT_V7-1":
        net = BmmtNetV7(exchange_type="and", exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[0, 1, 2, 3])
    elif args.net_G == "BmmtNet_NAT_V7-2":
        net = BmmtNetV7(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="DINA",
                        freature_cross_layer=[0, 1, 2, 3])
    elif args.net_G == "BmmtNet_NAT_V7-3":
        net = BmmtNetV7(exchange_type="and", exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[])
    elif args.net_G == "BmmtNet_NAT_V7-4":
        net = BmmtNetV7(exchange_type="single", exchange_layer=[0, 1, 2, 3], feature_cross_type="NA",
                        freature_cross_layer=[])
    elif args.net_G == "BmmtNet_NAT_V7-5":
        net = BmmtNetV7(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="SA",
                        freature_cross_layer=[0, 1, 2, 3])
    elif args.net_G == "BmmtNet_NAT_V7-6":
        net = BmmtNetV7(exchange_type=None, exchange_layer=[0, 1, 2, 3], feature_cross_type="DINA",
                        freature_cross_layer=[1, 2, 3])
    elif args.net_G == "BmmtNet_res34":
        net =BmmtNet_Res34()
    elif args.net_G == "BmmtNet_segformer_b2":
        net = BmmtNet_segformer_b2()
    elif args.net_G == "BmmtNet_segformer_b1":
        net = BmmtNet_segformer_b1()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# main Functions
###############################################################################


class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False, True, True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x


class BASE_Transformer(ResNet):
    """
    Resnet of 8 downsampling + BIT + bitemporal feature Differencing + a small CNN
    """

    def __init__(self, input_nc, output_nc, with_pos, resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True):
        super(BASE_Transformer, self).__init__(input_nc, output_nc, backbone=backbone,
                                               resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(32, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size

        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = 32
        mlp_dim = 2 * dim

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, 32))
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                                  decoder_pos_size,
                                                                  decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=dim, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0,
                                                      softmax=decoder_softmax)

    def _forward_semantic_tokens(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _forward_transformer_decoder(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def forward(self, x1, x2):
        # forward backbone resnet
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)

        #  forward tokenzier
        if self.tokenizer:
            token1 = self._forward_semantic_tokens(x1)
            token2 = self._forward_semantic_tokens(x2)
        else:
            token1 = self._forward_reshape_tokens(x1)
            token2 = self._forward_reshape_tokens(x2)
        # forward transformer encoder
        if self.token_trans:
            self.tokens_ = torch.cat([token1, token2], dim=1)
            self.tokens = self._forward_transformer(self.tokens_)
            token1, token2 = self.tokens.chunk(2, dim=1)
        # forward transformer decoder
        if self.with_decoder:
            x1 = self._forward_transformer_decoder(x1, token1)
            x2 = self._forward_transformer_decoder(x2, token2)
        else:
            x1 = self._forward_simple_decoder(x1, token1)
            x2 = self._forward_simple_decoder(x2, token2)
        # feature differencing
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        # forward small cnn
        x = self.classifier(x)
        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    model = CDFAModel(embed_dim=64).cuda()  # STA
    model=model.cuda()
    dummy_input = torch.randn(1, 3, 256, 256).cuda()



    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())

        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

        return {'Total': total_num / 1000000.0, 'Trainable': trainable_num / 1000000.0}


    # 查看网络参数

    print("params:",get_parameter_number(model))


    print("--------------thop--------------")
    from thop import profile, clever_format
    #thop:
    MACs, params = profile(model, (dummy_input, dummy_input))
    print('MACs: ', MACs, 'params: ', params)
    print('thop :MACs: %.2f G, params: %.2f M' % (MACs / 10 ** 9, params / 1000000.0))  # k（千）、M（百万）、G（十亿）、T（万亿）




