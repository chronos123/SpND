import torch
import torchvision.ops
from torch import nn


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x


class DeformableConv2dNorm(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):

        offset = self.offset_conv(x)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                        #   mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x


"""
author: sxc
function for calculating the coordintes of the kernel
"""

import numpy as np
from numpy import sin, cos, tan, pi, arcsin, arctan, arctan2
from functools import lru_cache
import torch
from torch import nn
from torch.nn.parameter import Parameter


@lru_cache(None)
def get_xy(delta_phi, delta_theta):
    return np.array([
        [
            (-tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1),
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
        ]
    ])
    

@torch.no_grad()
def get_xy_torch(delta_phi, delta_theta, device):
    return torch.FloatTensor([
        [
            (-torch.tan(delta_theta), 1/torch.cos(delta_theta)*torch.tan(delta_phi)),
            (0, torch.tan(delta_phi)),
            (torch.tan(delta_theta), 1/torch.cos(delta_theta)*torch.tan(delta_phi)),
        ],
        [
            (-torch.tan(delta_theta), 0),
            (1, 1),
            (torch.tan(delta_theta), 0),
        ],
        [
            (-torch.tan(delta_theta), -1/torch.cos(delta_theta)*torch.tan(delta_phi)),
            (0, -torch.tan(delta_phi)),
            (torch.tan(delta_theta), -1/torch.cos(delta_theta)*torch.tan(delta_phi)),
        ]
    ]).to(device)

@lru_cache(None)
def cal_index(h, w, img_r, img_c):
    phi = -((img_r+0.5)/h*pi - pi/2)
    theta = (img_c+0.5)/w*2*pi-pi

    delta_phi = pi/h
    delta_theta = 2*pi/w

    xys = get_xy(delta_phi, delta_theta)
    x = xys[..., 0]
    y = xys[..., 1]
    rho = np.sqrt(x**2+y**2)
    v = arctan(rho)
    new_phi= arcsin(cos(v)*sin(phi) + y*sin(v)*cos(phi)/rho)
    new_theta = theta + arctan(x*sin(v) / (rho*cos(phi)*cos(v) - y*sin(phi)*sin(v)))
    new_r = (-new_phi+pi/2)*h/pi - 0.5
    new_c = (new_theta+pi)*w/2/pi - 0.5
    new_c = (new_c + w) % w
    new_result = np.stack([new_r, new_c], axis=-1)
    new_result[1, 1] = (img_r, img_c)
    return new_result


@lru_cache(None)
def cal_index_torch(h, w, img_r, img_c, device):
    with torch.no_grad():
        phi = torch.FloatTensor([-((img_r+0.5)/h*torch.pi - torch.pi/2)]).to(device)
        theta = torch.FloatTensor([(img_c+0.5)/w*2*torch.pi-torch.pi]).to(device)

        delta_phi = torch.FloatTensor([torch.pi/h]).to(device)
        delta_theta = torch.FloatTensor([2*torch.pi/w]).to(device)

        xys = get_xy_torch(delta_phi, delta_theta, device)
        x = xys[..., 0]
        y = xys[..., 1]
        rho = torch.sqrt(x**2+y**2).to(device)
        v = torch.arctan(rho).to(device)
        new_phi= torch.arcsin(torch.cos(v)*torch.sin(phi) + y*torch.sin(v)*torch.cos(phi)/rho).to(device)
        new_theta = theta + torch.arctan(x*torch.sin(v) / (rho*torch.cos(phi)*torch.cos(v) - y*torch.sin(phi)*torch.sin(v))).to(device)
        new_r = (-new_phi+torch.pi/2)*h/torch.pi - 0.5
        new_c = (new_theta+torch.pi)*w/2/torch.pi - 0.5
        new_c = (new_c + w) % w
        new_r = new_r.cpu().numpy()
        new_c = new_c.cpu().numpy()
        new_result = np.stack([new_r, new_c], axis=-1)
        new_result[1, 1] = (img_r, img_c)
        return new_result

@lru_cache(None)
def _gen_filters_coordinates(h, w, stride):
    co = np.array([[cal_index(h, w, i, j) for j in range(0, w, stride)] for i in range(0, h, stride)])
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


@torch.no_grad()
def _gen_filters_coordinates_torch(h, w, stride, device):
    co = np.array([[cal_index_torch(h, w, i, j, device) for j in range(0, w, stride)] for i in range(0, h, stride)])
    co = co.transpose([4, 0, 1, 2, 3])
    co = torch.from_numpy(co).to(device)
    return co.contiguous()


def gen_filters_coordinates(h, w, stride=1):
    assert(isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates(h, w, stride).copy()

@torch.no_grad()
def gen_filters_coordinates_torch(h, w, stride=1, device="cpu"):
    assert(isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates_torch(h, w, stride, device).clone()


def gen_grid_coordinates(h, w, stride=1):
    coordinates = gen_filters_coordinates(h, w, stride).copy()
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1
    coordinates = coordinates[::-1]
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0]*sz[1], sz[2]*sz[3], sz[4])

    return coordinates.copy()


@torch.no_grad()
def gen_grid_coordinates_torch(h, w, stride=1, device="cpu"):
    coordinates = gen_filters_coordinates_torch(h, w, stride, device=device)
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1
    coordinates = torch.flip(coordinates, dims=[0])

    coordinates = coordinates.permute(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.contiguous().view(1, sz[0]*sz[1], sz[2]*sz[3], sz[4])

    return coordinates


class SphereDeformableConv2d(nn.Module):
    # no modulate - Deformable Conv (https://arxiv.org/abs/1703.06211)
    # author: sxc
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super().__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        off_padding = 1 if self.padding == 0 else self.padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=off_padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        max_offset = max(h, w)/50.

        initial_offset = self.create_sphere_offset_torch(h, w, device=x.device)
        self.initial_offset = initial_offset.repeat(b, 1, 1, 1)

        offset = self.offset_conv(x).clamp(-max_offset, max_offset) +  self.initial_offset.to(x.device)  # .clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=None,
                                          stride=self.stride,
                                          dilation=self.dilation
                                          )
        return x
    
    def create_sphere_offset(self, h, w):
        assert self.kernel_size == (3, 3), "sphereNet initialisation only supports 3x3 kernel now"
        assert self.stride[0] == self.stride[1], "stride should be same along height and width"
        coordinates = gen_grid_coordinates(h, w, self.stride[0])

        coordinates_x = torch.from_numpy(coordinates[0, :, :, 1].astype(np.float32))
        coordinates_y = torch.from_numpy(coordinates[0, :, :, 0].astype(np.float32))

        offset = torch.zeros((2 * self.kernel_size[0] * self.kernel_size[1], h, w))

        partition_pattern_0 = np.arange(0, coordinates.shape[1], self.kernel_size[0])
        partition_pattern_1 = np.arange(1, coordinates.shape[1], self.kernel_size[0])
        partition_pattern_2 = np.arange(2, coordinates.shape[1], self.kernel_size[0])
        
        w_partition_pattern_0 = np.arange(0, coordinates.shape[2], self.kernel_size[0])
        w_partition_pattern_1 = np.arange(1, coordinates.shape[2], self.kernel_size[0])
        w_partition_pattern_2 = np.arange(2, coordinates.shape[2], self.kernel_size[0])

        offset[1, :, :] = coordinates_x[partition_pattern_0][:, w_partition_pattern_2]
        offset[0, :, :] = coordinates_y[partition_pattern_0][:, w_partition_pattern_2]

        offset[3, :, :] = coordinates_x[partition_pattern_1][:, w_partition_pattern_2]
        offset[2, :, :] = coordinates_y[partition_pattern_1][:, w_partition_pattern_2]
        
        offset[5, :, :] = coordinates_x[partition_pattern_2][:, w_partition_pattern_2]
        offset[4, :, :] = coordinates_y[partition_pattern_2][:, w_partition_pattern_2]
        
        offset[7, :, :] = coordinates_x[partition_pattern_0][:, w_partition_pattern_1]
        offset[6, :, :] = coordinates_y[partition_pattern_0][:, w_partition_pattern_1]
        
        # center
        offset[9, :, :] = coordinates_x[partition_pattern_1][:, w_partition_pattern_1]
        offset[8, :, :] = coordinates_y[partition_pattern_1][:, w_partition_pattern_1]
        
        offset[11, :, :] = coordinates_x[partition_pattern_2][:, w_partition_pattern_1]
        offset[10, :, :] = coordinates_y[partition_pattern_2][:, w_partition_pattern_1]
        
        offset[13, :, :] = coordinates_x[partition_pattern_0][:, w_partition_pattern_0]
        offset[12, :, :] = coordinates_y[partition_pattern_0][:, w_partition_pattern_0]
        
        offset[15, :, :] = coordinates_x[partition_pattern_1][:, w_partition_pattern_0]
        offset[14, :, :] = coordinates_y[partition_pattern_1][:, w_partition_pattern_0]
        
        offset[17, :, :] = coordinates_x[partition_pattern_2][:, w_partition_pattern_0]
        offset[16, :, :] = coordinates_y[partition_pattern_2][:, w_partition_pattern_0]

        h_partition = np.arange(0, 18, 2)
        w_partition = np.arange(1, 18, 2)

        offset[w_partition, :, :] = self.unnorm(offset[w_partition, :, :], w) % w
        offset[h_partition, :, :] = self.unnorm(offset[h_partition, :, :], h)
        
        grid_y = torch.arange(0, h)
        grid_x = torch.arange(0, w)

        return offset

    def create_sphere_offset_torch(self, h, w, device):
        assert self.kernel_size == (3, 3), "sphereNet initialisation only supports 3x3 kernel now"
        assert self.stride[0] == self.stride[1], "stride should be same along height and width"
        coordinates = gen_grid_coordinates_torch(h, w, self.stride[0], device=device)
        coordinates_x = coordinates[0, :, :, 1]
        coordinates_y = coordinates[0, :, :, 0]
        offset = torch.zeros((2 * self.kernel_size[0] * self.kernel_size[1], h, w))

        partition_pattern_0 = torch.arange(0, coordinates.shape[1], self.kernel_size[0])
        partition_pattern_1 = torch.arange(1, coordinates.shape[1], self.kernel_size[0])
        partition_pattern_2 = torch.arange(2, coordinates.shape[1], self.kernel_size[0])
        
        w_partition_pattern_0 = torch.arange(0, coordinates.shape[2], self.kernel_size[0])
        w_partition_pattern_1 = torch.arange(1, coordinates.shape[2], self.kernel_size[0])
        w_partition_pattern_2 = torch.arange(2, coordinates.shape[2], self.kernel_size[0])

        offset[1, :, :] = coordinates_x[partition_pattern_0][:, w_partition_pattern_2]
        offset[0, :, :] = coordinates_y[partition_pattern_0][:, w_partition_pattern_2]

        offset[3, :, :] = coordinates_x[partition_pattern_1][:, w_partition_pattern_2]
        offset[2, :, :] = coordinates_y[partition_pattern_1][:, w_partition_pattern_2]
        
        offset[5, :, :] = coordinates_x[partition_pattern_2][:, w_partition_pattern_2]
        offset[4, :, :] = coordinates_y[partition_pattern_2][:, w_partition_pattern_2]
        
        offset[7, :, :] = coordinates_x[partition_pattern_0][:, w_partition_pattern_1]
        offset[6, :, :] = coordinates_y[partition_pattern_0][:, w_partition_pattern_1]

        offset[9, :, :] = coordinates_x[partition_pattern_1][:, w_partition_pattern_1]
        offset[8, :, :] = coordinates_y[partition_pattern_1][:, w_partition_pattern_1]
        
        offset[11, :, :] = coordinates_x[partition_pattern_2][:, w_partition_pattern_1]
        offset[10, :, :] = coordinates_y[partition_pattern_2][:, w_partition_pattern_1]
        
        offset[13, :, :] = coordinates_x[partition_pattern_0][:, w_partition_pattern_0]
        offset[12, :, :] = coordinates_y[partition_pattern_0][:, w_partition_pattern_0]
        
        offset[15, :, :] = coordinates_x[partition_pattern_1][:, w_partition_pattern_0]
        offset[14, :, :] = coordinates_y[partition_pattern_1][:, w_partition_pattern_0]
        
        offset[17, :, :] = coordinates_x[partition_pattern_2][:, w_partition_pattern_0]
        offset[16, :, :] = coordinates_y[partition_pattern_2][:, w_partition_pattern_0]

        h_partition = torch.arange(0, 18, 2)
        w_partition = torch.arange(1, 18, 2)

        offset[w_partition, :, :] = self.unnorm(offset[w_partition, :, :], w) % w
        offset[h_partition, :, :] = self.unnorm(offset[h_partition, :, :], h)

        return offset

    def unnorm(self, offsets, length):
        return offsets * length/2
   
    def extract_offset(self, x):
        b, c, h, w = x.shape
        max_offset = max(h, w)/50.
        initial_offset = self.create_sphere_offset_torch(h, w, device=x.device)
        self.initial_offset = initial_offset.repeat(b, 1, 1, 1)
        offset_deform = self.offset_conv(x).clamp(-max_offset, max_offset)
        offset = offset_deform +  self.initial_offset.to(x.device)
        return offset
