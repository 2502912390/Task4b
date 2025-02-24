import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.layers.helpers import to_2tuple

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

class attention2d(nn.Module):
    def __init__(self, in_planes, kernel_size, stride, padding, n_basis_kernels, temperature, pool_dim):
        super(attention2d, self).__init__()
        self.pool_dim = pool_dim
        self.temperature = temperature

        hidden_planes = int(in_planes / 4)

        if hidden_planes < 4:
            hidden_planes = 4

        if not pool_dim == 'both':
            self.conv1d1 = nn.Conv1d(in_planes, hidden_planes, kernel_size, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm1d(hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv1d2 = nn.Conv1d(hidden_planes, n_basis_kernels, 1, bias=True)
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            self.fc1 = nn.Linear(in_planes, hidden_planes)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(hidden_planes, n_basis_kernels)

    def forward(self, x): #x size : [bs, chan, frames, freqs]
        if self.pool_dim == 'freq':
            x = torch.mean(x, dim=3)  #x size : [bs, chan, frames]
        elif self.pool_dim == 'time':
            x = torch.mean(x, dim=2)  #x size : [bs, chan, freqs]
        elif self.pool_dim == 'both':
            # x = torch.mean(torch.mean(x, dim=2), dim=1)  #x size : [bs, chan]
            x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        elif self.pool_dim == 'chan':
            x = torch.mean(x, dim=1)  #x size : [bs, freqs, frames]

        if not self.pool_dim == 'both':
            x = self.conv1d1(x)               #x size : [bs, hid_chan, frames]
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv1d2(x)               #x size : [bs, n_ker, frames]
        else:
            x = self.fc1(x)               #x size : [bs, hid_chan]
            x = self.relu(x)
            x = self.fc2(x)               #x size : [bs, n_ker]

        return F.softmax(x / self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False, n_basis_kernels=4,
                 temperature=31, pool_dim='freq'):
        super(Dynamic_conv2d, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_dim = pool_dim

        self.n_basis_kernels = n_basis_kernels
        self.attention = attention2d(in_planes, self.kernel_size, self.stride, self.padding, n_basis_kernels,
                                     temperature, pool_dim)

        self.weight = nn.Parameter(torch.randn(n_basis_kernels, out_planes, in_planes, self.kernel_size, self.kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_basis_kernels, out_planes))
        else:
            self.bias = None

        for i in range(self.n_basis_kernels):
            nn.init.kaiming_normal_(self.weight[i])

    def forward(self, x): #x size : [bs, in_chan, frames, freqs]
        if self.pool_dim in ['freq', 'chan']:
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(4)    # size : [bs, n_ker, 1, frames, 1]
        elif self.pool_dim == 'time':
            softmax_attention = self.attention(x).unsqueeze(2).unsqueeze(3)    # size : [bs, n_ker, 1, 1, freqs]
        elif self.pool_dim == 'both':
            softmax_attention = self.attention(x).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)    # size : [bs, n_ker, 1, 1, 1]

        batch_size = x.size(0)

        aggregate_weight = self.weight.view(-1, self.in_planes, self.kernel_size, self.kernel_size) # size : [n_ker * out_chan, in_chan]

        if self.bias is not None:
            aggregate_bias = self.bias.view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding)
            # output size : [bs, n_ker * out_chan, frames, freqs]

        output = output.view(batch_size, self.n_basis_kernels, self.out_planes, output.size(-2), output.size(-1))
        # output size : [bs, n_ker, out_chan, frames, freqs]

        if self.pool_dim in ['freq', 'chan']:
            assert softmax_attention.shape[-2] == output.shape[-2]
        elif self.pool_dim == 'time':
            assert softmax_attention.shape[-1] == output.shape[-1]

        output = torch.sum(output * softmax_attention, dim=1)  # output size : [bs, out_chan, frames, freqs]

        return output

class CNN(nn.Module):
    def __init__(self,
                 n_in_channel,
                 activation="Relu",
                 conv_dropout=0,
                 kernel_size=[3, 3, 3],
                 padding=[1, 1, 1],
                 stride=[1, 1, 1],
                 nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 normalization="batch",
                 n_basis_kernels=4,
                 DY_layers=[0, 1, 1],#第一层普通卷积
                 temperature=31,
                 pool_dim='freq',
                 **transformer_kwargs):
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        self.n_filt_last = nb_filters[-1]
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ='relu'):
            in_dim = n_in_channel if i == 0 else nb_filters[i - 1]
            out_dim = nb_filters[i]
            if DY_layers[i] == 1:
                # cnn.add_module("conv{0}".format(i), Dynamic_conv2d(in_dim, out_dim, kernel_size[i], stride[i], padding[i],
                #                                                    n_basis_kernels=n_basis_kernels,
                #                                                    temperature=temperature, pool_dim=pool_dim))
                cnn.add_module("conv{0}".format(i),FDYFormerBlock(in_planes=in_dim,next_chan=out_dim,
                                                                   token_mixer=Dynamic_conv2d(in_planes=in_dim, out_planes=out_dim,
                                                                   kernel_size=kernel_size[i], stride=stride[i], padding=padding[i],
                                                                   n_basis_kernels=4,temperature=temperature, pool_dim=pool_dim)))
                # print("FDYFormer")
            else:
                cnn.add_module("conv{0}".format(i), nn.Conv2d(in_dim, out_dim, kernel_size[i], stride[i], padding[i]))

            if normalization == "batch":
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(out_dim, eps=0.001, momentum=0.99))
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, out_dim))

            if activ.lower() == "leakyrelu":
                cnn.add_module("Relu{0}".format(i), nn.LeakyReLu(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("Relu{0}".format(i), nn.ReLu())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(out_dim))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(out_dim))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))
        self.cnn = cnn

    def forward(self, x):    #x size : [bs, chan, frames, freqs]
        x = self.cnn(x)
        return x


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """

    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class FDYFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, in_planes,next_chan,token_mixer,
                 mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()

        self.token_mixer = token_mixer

        self.norm1 = norm_layer(in_planes)#经过token_mixer后 通道改变了，所以接下来的组件都要使用next_chan

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=next_chan, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=next_chan, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(next_chan)
        self.mlp = mlp(dim=next_chan, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=next_chan, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=next_chan, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x=self.norm1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.token_mixer(x)
        x = x.permute(0, 2, 3, 1)

        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    x
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        x = x.permute(0, 3, 1, 2)
        return x

# if __name__ == '__main__':
#     # (60 626 128 1)
#     x = torch.rand(4, 1, 626,128)
#
#     #MetaFormerBlock不能对通道进行改变
#     former = FDYFormerBlock(in_planes=1,next_chan=32, token_mixer=Dynamic_conv2d(in_planes=1, out_planes=32, kernel_size=3, stride=1, padding=1, bias=False, n_basis_kernels=4,
#                    temperature=31, pool_dim='freq'))
#
#     output = former(x)
#
#     print(x.size())
#     print(output.size())

import yaml
import os
if __name__ == '__main__':
    file_path = os.path.join('..', '..', 'recipes', 'dcase2024_task4_baseline', 'confs', 'pretrained.yaml')

    with open(file_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    input = torch.randn(4,1,626,128)
    cnn=CNN(**configs["net"])
    output = cnn(input)

    print(input.shape)
    print(output.shape)
