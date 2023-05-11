"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from torchkit import pytorch_utils as ptu
from torchkit.core import PyTorchModule
from torchkit.modules import LayerNorm
import torchcde
import torchsde
import pdb

relu_name = "relu"
elu_name = "elu"
ACTIVATIONS = {
    relu_name: nn.ReLU,
    elu_name: nn.ELU,
}


class Mlp(PyTorchModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=ptu.identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along last dim
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_shape,
        embed_size=100,
        depths=[8, 16],
        kernel_size=2,
        stride=1,
        activation=relu_name,
        from_flattened=False,
        normalize_pixel=False,
    ):
        super(ImageEncoder, self).__init__()
        self.shape = image_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.depths = [image_shape[0]] + depths

        layers = []
        h_w = self.shape[-2:]

        for i in range(len(self.depths) - 1):
            layers.append(
                nn.Conv2d(self.depths[i], self.depths[i + 1], kernel_size, stride)
            )
            layers.append(ACTIVATIONS[activation]())
            h_w = conv_output_shape(h_w, kernel_size, stride)

        self.cnn = nn.Sequential(*layers)

        self.linear = nn.Linear(
            h_w[0] * h_w[1] * self.depths[-1], embed_size
        )  # dreamer does not use it

        self.from_flattened = from_flattened
        self.normalize_pixel = normalize_pixel
        self.embed_size = embed_size

    def forward(self, image):
        # return embedding of shape [N, embed_size]
        if self.from_flattened:
            # image of size (T, B, C*H*W)
            batch_size = image.shape[:-1]
            img_shape = [np.prod(batch_size)] + list(self.shape)  # (T*B, C, H, W)
            image = torch.reshape(image, img_shape)
        else:  # image of size (N, C, H, W)
            batch_size = [image.shape[0]]

        if self.normalize_pixel:
            image = image / 255.0

        embed = self.cnn(image)  # (T*B, C, H, W)

        embed = torch.reshape(embed, list(batch_size) + [-1])  # (T, B, C*H*W)
        embed = self.linear(embed)  # (T, B, embed_size)
        return embed

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, width=128):
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.activation = F.tanh
        self.linear0 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.linear1 = torch.nn.Linear(hidden_channels, width)
        self.linear2 = torch.nn.Linear(width, input_channels * hidden_channels)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear0(z)
        
        z = z.relu()
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        
        z = self.activation(z)
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
       
        return z

    
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, 
                  width=128, radii=40):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels, width=width)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.radii=radii

    def forward(self, coeffs,init_hid=None):
        
        X = torchcde.LinearInterpolation(coeffs)
        
        if init_hid==None:
            X0= X.evaluate(X.interval[0])	
            z0 = self.initial(X0)
            z0_norms= torch.norm(z0,dim=1)**(-1)
            z0 = self.radii* z0_norms.unsqueeze(1).expand(z0.size(0), z0.size(1)) * z0
            #print(torch.norm(z0,dim=0))
        else:	   
            z0 = init_hid
        
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.grid_points,adjoint=False, backend= "torchdiffeq", atol = 0.001, rtol =0.001, method = "rk4")

        #pdb.set_trace()
        pred_y = self.readout(z_T)
        return pred_y, z_T
