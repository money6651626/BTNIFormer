import torch.nn.functional as F
from einops import repeat, reduce, rearrange
import warnings
from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from natten import NeighborhoodAttention2D as NeighborhoodAttention
from natten import NeighborhoodAttention2D_cross as NeighborhoodAttention_cross
from model.basemodel.STANET.mynet3 import build_backbone

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialExchange(nn.Module):
    """
    spatial exchange
    Args:
        p (float, optional): p of the features will be exchanged.
            Defaults to 1/2.
    """

    def __init__(self, p=2, model=None,rerange=False):
        super().__init__()
        assert p >= 1
        self.p = p
        self.model = model
        self.rerange=rerange
    def forward(self, x1, x2):

        if self.model is None:
            return x1, x2
        if self.rerange:
            x1=rearrange(x1,'b h w c -> b c h w')
            x2 = rearrange(x2, 'b h w c -> b c h w')
        N ,c, h, w = x1.shape
        # 创建掩码矩阵
        exchange_mask_w = (torch.arange(w) % self.p == 0).expand(h, w)
        exchange_mask_h = (torch.arange(h) % self.p == 0).expand(h, w).T
        if self.model == "and":
            exchange_mask = exchange_mask_w & exchange_mask_h
        elif self.model == "or":
            exchange_mask = exchange_mask_w | exchange_mask_h
        elif self.model == "single":
            exchange_mask = exchange_mask_w
        #print(exchange_mask)
        #print((exchange_mask == True).sum() / (h * w))
        # 创建输出张量并进行掩码替换
        exchange_mask=exchange_mask.cuda()
        out_x1 = torch.where(exchange_mask, x1, x2)
        out_x2 = torch.where(~exchange_mask, x1, x2)

        if self.rerange:
            out_x1 = rearrange(out_x1, 'b c h w -> b h w c')
            out_x2 = rearrange(out_x2, 'b c h w -> b h w c')
        return out_x1, out_x2




def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


# Transformer Decoder
class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = rearrange(x, 'b n h w->b (h w) n')
        x = self.proj(x)
        return x


# Difference module
def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )


# Intermediate prediction module
def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = torch.add(self.conv2(out) * 0.1, x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        #         reflection_padding = kernel_size // 2
        #         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        #         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


class Cross_dif(nn.Module):
    def __init__(self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):

        _, x_h, x_w,_ = x1.shape

        x1 = rearrange(x1, 'b h w c -> b (h w) c')
        x2 = rearrange(x2, 'b h w c-> b (h w) c')

        b, n, _, h = *x1.shape, self.num_heads
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x2)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), [q, k, v])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj_drop(self.proj(out))+x2
        out = rearrange(out, 'b (h w) c -> b  h w c', h=x_h, w=x_w)
        return out


class DecoderTransformer(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[64, 128, 256, 512], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16]):
        super(DecoderTransformer, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)


        # convolutional Difference Modules

        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)


        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):

        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []

        # Stage 4: x1/32 scale

        _c4_1 = rearrange(self.linear_c4(c4_1), 'b (h w) n-> b n h w', h=c4_1.shape[2], w=c4_1.shape[3])  # 投影到embed维度上
        _c4_2 = rearrange(self.linear_c4(c4_2), 'b (h w) n-> b n h w', h=c4_2.shape[2], w=c4_2.shape[3])
        # Stage 3: x1/16 scale
        _c3_1 = rearrange(self.linear_c3(c3_1), 'b (h w) n-> b n h w', h=c3_1.shape[2], w=c3_1.shape[3])
        _c3_2 = rearrange(self.linear_c3(c3_2), 'b (h w) n-> b n h w', h=c3_2.shape[2], w=c3_2.shape[3])
        # Stage 2: x1/8 scale
        _c2_1 = rearrange(self.linear_c2(c2_1), 'b (h w) n-> b n h w', h=c2_1.shape[2], w=c2_1.shape[3])
        _c2_2 = rearrange(self.linear_c2(c2_2), 'b (h w) n-> b n h w', h=c2_2.shape[2], w=c2_2.shape[3])
        # Stage 1: x1/4 scale
        _c1_1 = rearrange(self.linear_c1(c1_1), 'b (h w) n-> b n h w', h=c1_1.shape[2], w=c1_1.shape[3])
        _c1_2 = rearrange(self.linear_c1(c1_2), 'b (h w) n-> b n h w', h=c1_2.shape[2], w=c1_2.shape[3])


        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1)) + F.interpolate(_c4, scale_factor=2, mode="bilinear")
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1)) + F.interpolate(_c3, scale_factor=2, mode="bilinear")
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1)) + F.interpolate(_c2, scale_factor=2, mode="bilinear")


        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        # _c1_up = resize(_c1, size=c1_2.size()[2:], mode='bilinear', align_corners=False)



        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)  # 转置卷积上采样一倍(h/2,w/2)
        # Residual block
        x = self.dense_2x(x)  # 残差块精炼信息
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)  # 转置卷积上采样一倍(h,w) 参数不同
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)  # 用一个卷积层压缩通道获取最终结果

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs


class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class NATLayer(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            kernel_size=7,
            dilation=None,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x):
        if not self.layer_scale:
            shortcut = x
            x = self.norm1(x)
            x = self.attn(x)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class NATBlock(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            num_heads,
            kernel_size,
            dilations=None,
            downsample=True,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            layer_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is None:
            return x, x
        return self.downsample(x), x

class Encoder_v1(nn.Module):
    #oral 就原生不做改动
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        frozen_stages=-1,
        pretrained=None,
        layer_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.init_weights(pretrained)



    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(xo)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)

    def forward_features(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)
class Encoder_v2(nn.Module):
    #特征提取过程中进行exchange，对于特征提取的多尺度结果进行crossattention
    def __init__(
            self,
            embed_dim,
            mlp_ratio,
            depths,
            num_heads,
            drop_path_rate=0.2,
            in_chans=3,
            kernel_size=7,
            dilations=None,
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            frozen_stages=-1,
            pretrained=None,
            layer_scale=None,
            exchange_type=None,
            exchange_layer=[],
            feature_cross_type="NA",
            freature_cross_layer=[],
            **kwargs,
    ):
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.exchange = SpatialExchange(model=exchange_type, p=2,rerange=True)
        self.exchange_layer=exchange_layer
        self.feature_cross_type=feature_cross_type
        self.freature_cross_layer=freature_cross_layer

        self.cross_feature_A=nn.ModuleList()
        self.cross_feature_B=nn.ModuleList()
        for i in range(self.num_levels):
            if self.feature_cross_type == "NA":
                cross_layer = NeighborhoodAttention_cross(
                    dim=int(embed_dim * 2 ** i),
                    num_heads=num_heads[i],
                    kernel_size=kernel_size,
                    dilation=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                )
            elif self.feature_cross_type == "DINA":
                cross_layer = NeighborhoodAttention_cross(
                    dim=int(embed_dim * 2 ** i),
                    num_heads=num_heads[i],
                    kernel_size=kernel_size,
                    dilation= 2**(self.num_levels-1-i),
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                )
            else:
                cross_layer = Cross_dif(
                    dim=int(embed_dim * 2 ** i),
                    num_heads=num_heads[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                )
            self.cross_feature_A.append(cross_layer)
            self.cross_feature_B.append(cross_layer)


        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.init_weights(pretrained)


    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x1,x2):
        outs_1 = []
        outs_2 = []
        for idx, level in enumerate(self.levels):
            x1, xo_1 = level(x1)
            x2, xo_2= level(x2)
            if idx in self.exchange_layer:
                x1, x2 = self.exchange(x1,x2)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out_1 = norm_layer(xo_1)
                x_out_2 = norm_layer(xo_2)

                if idx in self.freature_cross_layer:

                    x_out_1_end=self.cross_feature_A[idx](x_out_2,x_out_1)
                    x_out_2 = self.cross_feature_B[idx](x_out_1, x_out_2)
                    x_out_1=x_out_1_end

                outs_1.append(rearrange(x_out_1,'B H W C-> B C H W '))
                outs_2.append(rearrange(x_out_2,'B H W C-> B C H W '))
        return outs_1,outs_2

    def forward(self, x1,x2):
        x1 = self.forward_embeddings(x1)
        x2 = self.forward_embeddings(x2)
        return self.forward_tokens(x1,x2)

    def forward_features(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)


class Encoder_v3(nn.Module):
    #在encoder的特征提取层加上exchange以及cross_attention (参与特征的提取过程)
    def __init__(
            self,
            embed_dim,
            mlp_ratio,
            depths,
            num_heads,
            drop_path_rate=0.2,
            in_chans=3,
            kernel_size=7,
            dilations=None,
            out_indices=(0, 1, 2, 3),
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=nn.LayerNorm,
            frozen_stages=-1,
            pretrained=None,
            layer_scale=None,
            exchange_type=None,
            exchange_layer=[],
            feature_cross_type="NA",
            freature_cross_layer=[],
            **kwargs,
    ):
        super().__init__()
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.exchange = SpatialExchange(model=exchange_type, p=2,rerange=True)
        self.exchange_layer=exchange_layer
        self.feature_cross_type=feature_cross_type
        self.freature_cross_layer=freature_cross_layer

        self.cross_feature_A=nn.ModuleList()
        self.cross_feature_B=nn.ModuleList()
        for i in range(self.num_levels):
            if self.feature_cross_type=="NA":
                cross_layer = NeighborhoodAttention_cross(
                    dim=int(embed_dim * 2 ** (i+1)),
                    num_heads=num_heads[i],
                    kernel_size=kernel_size,
                    dilation=1,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                )
            else:
                cross_layer = Cross_dif(
                    dim=int(embed_dim * 2 ** i),
                    num_heads=num_heads[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop_rate,
                )
            self.cross_feature_A.append(cross_layer)
            self.cross_feature_B.append(cross_layer)


        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
            )
            self.levels.append(level)

        # add a norm layer for each output
        self.out_indices = out_indices
        for i_layer in self.out_indices:
            layer = norm_layer(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        if pretrained is not None:
            self.init_weights(pretrained)


    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x1,x2):
        outs_1 = []
        outs_2 = []
        for idx, level in enumerate(self.levels):
            x1, xo_1 = level(x1)#输出的特征层是xo_1，前者是下采样的结果用于下一部提取
            x2, xo_2= level(x2)
            if idx in self.exchange_layer:
                x1, x2 = self.exchange(x1,x2)
            if idx in self.freature_cross_layer:
                x1_end = self.cross_feature_A[idx](x2, x1)
                x2 = self.cross_feature_B[idx](x1, x2)
                x1=x1_end

            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out_1 = norm_layer(xo_1)
                x_out_2 = norm_layer(xo_2)

                outs_1.append(rearrange(x_out_1,'B H W C-> B C H W '))
                outs_2.append(rearrange(x_out_2,'B H W C-> B C H W '))
        return outs_1,outs_2

    def forward(self, x1,x2):
        x1 = self.forward_embeddings(x1)
        x2 = self.forward_embeddings(x2)
        return self.forward_tokens(x1,x2)

    def forward_features(self, x):
        x = self.forward_embeddings(x)
        return self.forward_tokens(x)

class Decoder_simple(nn.Module):
    """
    Transformer Decoder
    """

    def __init__(self, input_transform='multiple_select', in_index=[0, 1, 2, 3], align_corners=True,
                 in_channels=[64, 128, 256, 512], embedding_dim=64, output_nc=2,
                 decoder_softmax=False, feature_strides=[2, 4, 8, 16]):
        super(Decoder_simple, self).__init__()
        # assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]

        # settings
        self.feature_strides = feature_strides
        self.input_transform = input_transform
        self.in_index = in_index
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.output_nc = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # MLP decoder heads
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=self.embedding_dim)


        # convolutional Difference Modules

        self.diff_c4 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c3 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c2 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)
        self.diff_c1 = conv_diff(in_channels=2 * self.embedding_dim, out_channels=self.embedding_dim)


        # Final linear fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=self.embedding_dim * len(in_channels), out_channels=self.embedding_dim,
                      kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.convd1x = UpsampleConvLayer(self.embedding_dim, self.embedding_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embedding_dim))
        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        # Final activation
        self.output_softmax = decoder_softmax
        self.active = nn.Sigmoid()

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs1, inputs2):

        # Transforming encoder features (select layers)
        x_1 = self._transform_inputs(inputs1)  # len=4, 1/2, 1/4, 1/8, 1/16
        x_2 = self._transform_inputs(inputs2)  # len=4, 1/2, 1/4, 1/8, 1/16

        # img1 and img2 features
        c1_1, c2_1, c3_1, c4_1 = x_1
        c1_2, c2_2, c3_2, c4_2 = x_2

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape

        outputs = []

        # Stage 4: x1/32 scale

        _c4_1 = rearrange(self.linear_c4(c4_1), 'b (h w) n-> b n h w', h=c4_1.shape[2], w=c4_1.shape[3])  # 投影到embed维度上
        _c4_2 = rearrange(self.linear_c4(c4_2), 'b (h w) n-> b n h w', h=c4_2.shape[2], w=c4_2.shape[3])
        # Stage 3: x1/16 scale
        _c3_1 = rearrange(self.linear_c3(c3_1), 'b (h w) n-> b n h w', h=c3_1.shape[2], w=c3_1.shape[3])
        _c3_2 = rearrange(self.linear_c3(c3_2), 'b (h w) n-> b n h w', h=c3_2.shape[2], w=c3_2.shape[3])
        # Stage 2: x1/8 scale
        _c2_1 = rearrange(self.linear_c2(c2_1), 'b (h w) n-> b n h w', h=c2_1.shape[2], w=c2_1.shape[3])
        _c2_2 = rearrange(self.linear_c2(c2_2), 'b (h w) n-> b n h w', h=c2_2.shape[2], w=c2_2.shape[3])
        # Stage 1: x1/4 scale
        _c1_1 = rearrange(self.linear_c1(c1_1), 'b (h w) n-> b n h w', h=c1_1.shape[2], w=c1_1.shape[3])
        _c1_2 = rearrange(self.linear_c1(c1_2), 'b (h w) n-> b n h w', h=c1_2.shape[2], w=c1_2.shape[3])


        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))
        _c1 = self.diff_c1(torch.cat((_c1_1, _c1_2), dim=1))


        _c4_up = resize(_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        _c3_up = resize(_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        _c2_up = resize(_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        # _c1_up = resize(_c1, size=c1_2.size()[2:], mode='bilinear', align_corners=False)



        # Linear Fusion of difference image from all scales
        _c = self.linear_fuse(torch.cat((_c4_up, _c3_up, _c2_up, _c1), dim=1))

        # #Dropout
        # if dropout_ratio > 0:
        #     self.dropout = nn.Dropout2d(dropout_ratio)
        # else:
        #     self.dropout = None

        # Upsampling x2 (x1/2 scale)
        x = self.convd2x(_c)  # 转置卷积上采样一倍(h/2,w/2)
        # Residual block
        x = self.dense_2x(x)  # 残差块精炼信息
        # Upsampling x2 (x1 scale)
        x = self.convd1x(x)  # 转置卷积上采样一倍(h,w) 参数不同
        # Residual block
        x = self.dense_1x(x)

        # Final prediction
        cp = self.change_probability(x)  # 用一个卷积层压缩通道获取最终结果

        outputs.append(cp)

        if self.output_softmax:
            temp = outputs
            outputs = []
            for pred in temp:
                outputs.append(self.active(pred))

        return outputs
class BmmtNetV1(torch.nn.Module):
    #原生未经过改动的NAT+changeformer
    def __init__(self, decoder_softmax=False, output_class=2, embed_dim=64, depths=[3, 4, 6, 5], exchange_type=None,feature_cross=False,
                 ):
        super(BmmtNetV1, self).__init__()
        self.embedim = embed_dim
        self.Tencoder = Encoder_v1(embed_dim=embed_dim,
                            mlp_ratio=3.0,
                            depths=depths,
                            num_heads=[2, 4, 8, 16],
                            drop_path_rate=0.2,
                            kernel_size=7,
                            dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]])
        self.Decoder = DecoderTransformer(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                             align_corners=False,
                                             in_channels=[embed_dim * 2 ** i for i in range(len(depths))],
                                             embedding_dim=64, output_nc=output_class,
                                             decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                             )

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tencoder(x1), self.Tencoder(x2)]
        cp = self.Decoder(fx1, fx2)
        return cp[-1]

class BmmtNetV2(torch.nn.Module):
    #在Encoder部分特征提取过程中进行exchange，对于特征提取的多尺度结果进行crossattention
    def __init__(self, decoder_softmax=False, output_class=2, embed_dim=64,depths=[3, 4, 6, 5], exchange_type=None,exchange_layer=[0,1,2,3],feature_cross_type="NA",freature_cross_layer=[0,1,2,3]
                 ):
        super(BmmtNetV2, self).__init__()
        self.embedim = embed_dim
        self.Tencoder = Encoder_v2(embed_dim=embed_dim,
                            mlp_ratio=3.0,
                            depths=depths,
                            num_heads=[2, 4, 8, 16],
                            drop_path_rate=0.2,
                            kernel_size=7,
                            dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],exchange_type=exchange_type,exchange_layer=exchange_layer,feature_cross_type=feature_cross_type,freature_cross_layer=freature_cross_layer)
        self.Decoder = DecoderTransformer(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                             align_corners=False,
                                             in_channels=[embed_dim * 2 ** i for i in range(len(depths))],
                                             embedding_dim=64, output_nc=output_class,
                                             decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                             )

    def forward(self, x1, x2):
        fx1, fx2 = self.Tencoder(x1,x2)
        cp = self.Decoder(fx1, fx2)
        return cp[-1]  # 8、16、32、64、256五个尺度的解码结果

class BmmtNetV3(torch.nn.Module):
    #在特征提取部分进行exchange，并在特征提取过程中赋予crossattention的信息
    def __init__(self, decoder_softmax=False, output_class=2, embed_dim=64,depths=[3, 4, 6, 5],exchange_type=None,exchange_layer=[0,1,2,3],feature_cross_type="NA",freature_cross_layer=[0,1,2],
                 ):
        super(BmmtNetV3, self).__init__()
        self.embedim = embed_dim
        self.Tencoder = Encoder_v3(embed_dim=embed_dim,
                            mlp_ratio=3.0,
                            depths=depths,
                            num_heads=[2, 4, 8, 16],
                            drop_path_rate=0.2,
                            kernel_size=7,
                            dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],exchange_type=exchange_type,exchange_layer=exchange_layer,feature_cross_type=feature_cross_type,freature_cross_layer=freature_cross_layer)
        self.Decoder = DecoderTransformer(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                             align_corners=False,
                                             in_channels=[embed_dim * 2 ** i for i in range(len(depths))],
                                             embedding_dim=64, output_nc=output_class,
                                             decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                             )

    def forward(self, x1, x2):
        fx1, fx2 = self.Tencoder(x1,x2)
        cp = self.Decoder(fx1, fx2)
        return cp[-1]  # 8、16、32、64、256五个尺度的解码结果

class BmmtNetV5(torch.nn.Module):
    #在特征提取部分进行exchange，并在特征提取过程中赋予crossattention的信息
    def __init__(self, decoder_softmax=False, output_class=2, embed_dim=64, depths=[3, 4, 6, 5],
                 ):
        super(BmmtNetV5, self).__init__()
        self.embedim = embed_dim
        self.Tencoder = Encoder_v1(embed_dim=embed_dim,
                            mlp_ratio=3.0,
                            depths=depths,
                            num_heads=[2, 4, 8, 16],
                            drop_path_rate=0.2,
                            kernel_size=7,
                            dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]])
        self.Decoder = Decoder_simple(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                          align_corners=False,
                                          in_channels=[embed_dim * 2 ** i for i in range(len(depths))],
                                          embedding_dim=64, output_nc=output_class,
                                          decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                          )

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tencoder(x1), self.Tencoder(x2)]
        cp = self.Decoder(fx1, fx2)
        return cp[-1]
class BmmtNetV6(torch.nn.Module):
    #在特征提取部分进行exchange，并在特征提取过程中赋予crossattention的信息
    def __init__(self, decoder_softmax=False, output_class=2, embed_dim=64, depths=[3, 4, 6, 5],
                 ):
        super(BmmtNetV6, self).__init__()
        self.embedim = embed_dim
        self.Tencoder = Encoder_v1(embed_dim=embed_dim,
                                   mlp_ratio=3.0,
                                   depths=depths,
                                   num_heads=[2, 4, 8, 16],
                                   drop_path_rate=0.2,
                                   kernel_size=7,
                                   dilations=[[1, 8, 1], [1, 2, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]])
        self.Decoder = Decoder_simple(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                          align_corners=False,
                                          in_channels=[embed_dim * 2 ** i for i in range(len(depths))],
                                          embedding_dim=64, output_nc=output_class,
                                          decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                          )

    def forward(self, x1, x2):
        [fx1, fx2] = [self.Tencoder(x1), self.Tencoder(x2)]
        cp = self.Decoder(fx1, fx2)
        return cp[-1]

class BmmtNetV7(torch.nn.Module):
    #在Encoder部分特征提取过程中进行exchange，对于特征提取的多尺度结果进行crossattention
    def __init__(self, decoder_softmax=False, output_class=2, embed_dim=64,depths=[3, 4, 6, 5], exchange_type=None,exchange_layer=[0,1,2,3],feature_cross_type="NA",freature_cross_layer=[0,1,2,3]
                 ):
        super(BmmtNetV7, self).__init__()
        self.embedim = embed_dim
        self.Tencoder = Encoder_v2(embed_dim=embed_dim,
                            mlp_ratio=3.0,
                            depths=depths,
                            num_heads=[2, 4, 8, 16],
                            drop_path_rate=0.2,
                            kernel_size=7,
                            dilations=[[1, 8, 1], [1, 2, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]],exchange_type=exchange_type,exchange_layer=exchange_layer,feature_cross_type=feature_cross_type,freature_cross_layer=freature_cross_layer)
        self.Decoder = Decoder_simple(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                          align_corners=False,
                                          in_channels=[embed_dim * 2 ** i for i in range(len(depths))],
                                          embedding_dim=64, output_nc=output_class,
                                          decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                          )

    def forward(self, x1, x2):
        fx1, fx2 = self.Tencoder(x1,x2)
        cp = self.Decoder(fx1, fx2)
        return cp[-1]  # 8、16、32、64、256五个尺度的解码结果



class BmmtNet_Res34(torch.nn.Module):
    #在Encoder部分特征提取过程中进行exchange，对于特征提取的多尺度结果进行crossattention
    def __init__(self, decoder_softmax=False, output_class=2, embed_dim=64,depths=[3, 4, 6, 5]
                 ):
        super(BmmtNet_Res34, self).__init__()
        self.embedim = embed_dim
        self.Tencoder = build_backbone(backbone='resnet34',in_c=3,output_stride=32,BatchNorm=nn.BatchNorm2d)
        self.Decoder = Decoder_simple(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                          align_corners=False,
                                          in_channels=[embed_dim * 2 ** i for i in range(len(depths))],
                                          embedding_dim=64, output_nc=output_class,
                                          decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                          )

    def forward(self, x1, x2):
        f4,f1,f2,f3=self.Tencoder(x1)
        out1=(f1,f2,f3,f4)
        f4, f1, f2, f3 = self.Tencoder(x2)
        out2 = (f1, f2, f3, f4)
        cp = self.Decoder(out1,out2)
        return cp[-1]  # 8、16、32、64、256五个尺度的解码结果


from model.basemodel.ChangeFormer import EncoderTransformer_v3


class BmmtNet_segformer_b2(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=64):
        super(BmmtNet_segformer_b2, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [3,3,6,3]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.backbone = EncoderTransformer_v3(img_size=256, patch_size=3, in_chans=input_nc, num_classes=output_nc,
                                             embed_dims=self.embed_dims,
                                             num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                             qk_scale=None, drop_rate=self.drop_rate,
                                             attn_drop_rate=self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             depths=self.depths, sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.Decoder = Decoder_simple(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                      align_corners=False,
                                      in_channels=self.embed_dims,
                                      embedding_dim=64, output_nc=output_nc,
                                      decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                      )

    def forward(self, x1, x2):
        [fx1, fx2] = [self.backbone(x1), self.backbone(x2)]

        cp = self.Decoder(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp[-1]


class BmmtNet_segformer_b1(nn.Module):

    def __init__(self, input_nc=3, output_nc=2, decoder_softmax=False, embed_dim=64):
        super(BmmtNet_segformer_b1, self).__init__()
        # Transformer Encoder
        self.embed_dims = [64, 128, 320, 512]
        self.depths = [2,2,2,2]
        self.embedding_dim = embed_dim
        self.drop_rate = 0.1
        self.attn_drop = 0.1
        self.drop_path_rate = 0.1

        self.backbone = EncoderTransformer_v3(img_size=256, patch_size=3, in_chans=input_nc, num_classes=output_nc,
                                             embed_dims=self.embed_dims,
                                             num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                                             qk_scale=None, drop_rate=self.drop_rate,
                                             attn_drop_rate=self.attn_drop, drop_path_rate=self.drop_path_rate,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             depths=self.depths, sr_ratios=[8, 4, 2, 1])

        # Transformer Decoder
        self.Decoder = Decoder_simple(input_transform='multiple_select', in_index=[0, 1, 2, 3],
                                      align_corners=False,
                                      in_channels=self.embed_dims,
                                      embedding_dim=64, output_nc=output_nc,
                                      decoder_softmax=decoder_softmax, feature_strides=[2, 4, 8, 16],
                                      )

    def forward(self, x1, x2):
        [fx1, fx2] = [self.backbone(x1), self.backbone(x2)]

        cp = self.Decoder(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        return cp[-1]
if __name__ == "__main__":
   pass