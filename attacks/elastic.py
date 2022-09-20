import random
import torch
import torch.nn as nn

def inverse_transform(x):
    x = x * 0.5 + 0.5
    return x * 255.

def transform(x):
    x = x / 255.
    return x * 2 - 1

class PixelModel(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        x = transform(x)
        x = self.model(x)
        return x

import math
import numbers
import numpy as np
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, inp):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(inp, weight=self.weight, groups=self.groups)


class ElasticDeformation(nn.Module):
    def __init__(self, im_size, filter_size, std):
        super().__init__()
        self.im_size = im_size
        self.filter_size = filter_size
        self.std = std
        self.kernel = GaussianSmoothing(2, self.filter_size, self.std).cuda()

        self._get_base_flow()

    def _get_base_flow(self):
        xflow, yflow = np.meshgrid(
                np.linspace(-1, 1, self.im_size, dtype='float32'),
                np.linspace(-1, 1, self.im_size, dtype='float32'))
        flow = np.stack((xflow, yflow), axis=-1)
        flow = np.expand_dims(flow, axis=0)
        self.base_flow = nn.Parameter(torch.from_numpy(flow)).cuda().detach()

    def warp(self, im, flow):
        return F.grid_sample(im, flow, mode='bilinear')

    def forward(self, im, params):
        flow = F.pad(params, ((self.filter_size - 1) // 2, ) * 4 , mode='reflect')
        local_flow = self.kernel(flow).transpose(1, 3).transpose(1, 2)
        return self.warp(im, local_flow + self.base_flow)

class ElasticAttackBase(object):
    def __init__(self, nb_its, eps_max, step_size, resol,
                 rand_init=True, scale_each=False,
                 kernel_size=25, kernel_std=3):
        '''
        Arguments:
            nb_its (int):          Number of iterations
            eps_max (float):       Maximum flow, in L_inf norm, in pixels
            step_size (float):     Maximum step size in L_inf norm, in pixels
            resol (int):           Side length of images, in pixels
            rand_init (bool):      Whether to do a random init
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            kernel_size (int):     Size, in pixels of gaussian kernel
            kernel_std (int):      Standard deviation of kernel
        '''
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each

        self.deformer = ElasticDeformation(resol, kernel_size, kernel_std)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its

    def _init(self, batch_size, eps):
        if self.rand_init:
            flow = torch.rand((batch_size, 2, self.resol, self.resol),
                              dtype=torch.float32, device='cuda') * 2 - 1
            flow = eps[:, None, None, None] * flow
        else:
            flow = torch.zeros((batch_size, 2, self.resol, self.resol),
                               dtype=torch.float32, device='cuda')
        flow.requires_grad_()
        return flow

    def _forward(self, pixel_model, pixel_img, target, scale_eps=False, avoid_target=True):
        pixel_inp = pixel_img.detach()
        pixel_inp.requires_grad = True

        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand * self.eps_max
            step_size = rand * self.step_size
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')

        base_eps.mul_(2.0 / self.resol)
        step_size.mul_(2.0 / self.resol)

        flow = self._init(pixel_img.size()[0], base_eps)
        pixel_inp_adv = self.deformer(pixel_inp, flow)

        if self.nb_its > 0:
            res = pixel_model(pixel_inp_adv)
            for it in range(self.nb_its):
                loss = self.criterion(res, target)
                loss.backward()

                if avoid_target:
                    grad = flow.grad.data
                else:
                    grad = -flow.grad.data

                flow.data = flow.data + step_size[:, None, None, None] * grad.sign()
                flow.data = torch.max(torch.min(flow.data, base_eps[:, None, None, None]),
                                      -base_eps[:, None, None, None])
                pixel_inp_adv = self.deformer(pixel_inp, flow)
                if it != self.nb_its - 1:
                    res = pixel_model(pixel_inp_adv)
                    flow.grad.data.zero_()
        return pixel_inp_adv


class ElasticAttack(object):
    def __init__(self,
                 predict,
                 nb_iters,
                 eps_max,
                 step_size,
                 resolution):
        self.pixel_model = PixelModel(predict)
        self.elastic_obj = ElasticAttackBase(
            nb_its=nb_iters,
            eps_max=eps_max,
            step_size=step_size,
            resol=resolution)

    def perturb(self, images, labels):
        pixel_img = inverse_transform(images.clamp(-1., 1.)).detach().clone()
        pixel_ret = self.elastic_obj._forward(
            pixel_model=self.pixel_model,
            pixel_img=pixel_img,
            target=labels)

        return transform(pixel_ret)

