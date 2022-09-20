import random
import numpy as np
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

def fog_creator(fog_vars, bsize=1, mapsize=256, wibbledecay=1.75):
    assert (mapsize & (mapsize - 1) == 0)
    maparray = torch.from_numpy(np.empty((bsize, mapsize, mapsize), dtype=np.float32)).cuda()
    maparray[:, 0, 0] = 0
    stepsize = mapsize
    wibble = 100

    var_num = 0

    def wibbledmean(array, var_num):
        result = array / 4. + fog_vars[var_num] * 2 * wibble - wibble
        return result

    def fillsquares(var_num):
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[:, 0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + torch.roll(cornerref, -1, 1)
        squareaccum = squareaccum + torch.roll(squareaccum, -1, 2)
        maparray[:, stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum, var_num)
        return var_num + 1

    def filldiamonds(var_num):
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.size(1)
        drgrid = maparray[:, stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[:, 0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + torch.roll(drgrid, 2, 1)
        lulsum = ulgrid + torch.roll(ulgrid, -1, 2)
        ltsum = ldrsum + lulsum
        maparray[:, 0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum, var_num)
        var_num += 1
        tdrsum = drgrid + torch.roll(drgrid, 2, 2)
        tulsum = ulgrid + torch.roll(ulgrid, -1, 1)
        ttsum = tdrsum + tulsum
        maparray[:, stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum, var_num)
        return var_num + 1

    while stepsize >= 2:
        var_num = fillsquares(var_num)
        var_num = filldiamonds(var_num)
        stepsize //= 2
        wibble /= wibbledecay

    maparray = maparray - maparray.min()
    return (maparray / maparray.max()).reshape(bsize, 1, mapsize, mapsize)

class FogAttackBase(object):
    def __init__(self, nb_its, eps_max, step_size, resol, rand_init=True, scale_each=False,
                 wibble_decay=2.0):
        """
        Parameters:
            nb_its (int):          Number of GD iterations.
            eps_max (float):       The max norm, in pixel space.
            step_size (float):     The max step size, in pixel space.
            resol (int):           Side length of the image.
            rand_init (bool):      Whether to init randomly in the norm ball
            scale_each (bool):     Whether to scale eps for each image in a batch separately
            wibble_decay (float):  Fog-specific parameter
        """
        self.nb_its = nb_its
        self.eps_max = eps_max
        self.step_size = step_size
        self.resol = resol
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.wibble_decay = wibble_decay

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.nb_backward_steps = self.nb_its

    def _init(self, batch_size, map_size):
        fog_vars = []
        for i in range(int(np.log2(map_size))):
            for j in range(3):
                var = torch.rand((batch_size, 2 ** i, 2 ** i), device="cuda")
                var.requires_grad_()
                fog_vars.append(var)
        return fog_vars

    def _forward(self, pixel_model, pixel_img, target, avoid_target=True, scale_eps=False):
        pixel_inp = pixel_img.detach()
        batch_size = pixel_img.size(0)
        x_max, _ = torch.max(pixel_img.view(pixel_img.size(0), 3, -1), -1)
        x_max = x_max.view(-1, 3, 1, 1)
        map_size = pixel_img.size()[2]     

        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_img.size()[0], device='cuda')
            else:
                rand = random.random() * torch.ones(pixel_img.size()[0], device='cuda')
            base_eps = rand.mul(self.eps_max)
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')
        else:
            base_eps = self.eps_max * torch.ones(pixel_img.size()[0], device='cuda')
            step_size = self.step_size * torch.ones(pixel_img.size()[0], device='cuda')

        fog_vars = self._init(batch_size, map_size)
        fog = fog_creator(fog_vars, batch_size, mapsize=map_size,
                          wibbledecay=self.wibble_decay)
        s = pixel_model(torch.clamp((pixel_inp + base_eps[:, None, None, None] * fog) /
                                    (x_max + base_eps[:, None, None, None]) * 255., 0., 255.))
        for it in range(self.nb_its):
            loss = self.criterion(s, target)
            loss.backward()
            '''
            Because of batching, this grad is scaled down by 1 / batch_size, which does not matter
            for what follows because of normalization.
            '''
            if avoid_target:
                grads = [f.grad.data for f in fog_vars]
            else:
                grads = [-f.grad.data for f in fog_vars]

            grad_signs = [grad.sign() for grad in grads]
            for f, g in zip(fog_vars, grad_signs):
                f.data = f.data + step_size[:, None, None] * g
                f.detach()
                f.data = f.data.clamp(0, 1)
                f.requires_grad_()

            if it != self.nb_its - 1:
                fog = fog_creator(fog_vars, batch_size, mapsize=map_size,
                                  wibbledecay=self.wibble_decay)
                s = pixel_model(torch.clamp((pixel_inp + base_eps[:, None, None, None] * fog) /
                                            (x_max + base_eps[:, None, None, None]) * 255., 0., 255.))
                for f in fog_vars:
                    f.grad.data.zero_()
        fog = fog_creator(fog_vars, batch_size, mapsize=map_size,
                          wibbledecay=self.wibble_decay)
        pixel_result = torch.clamp((pixel_inp + base_eps[:, None, None, None] * fog) /
                                   (x_max + base_eps[:, None, None, None]) * 255., 0., 255.)
        return pixel_result

class FogAttack(object):
    def __init__(self,
                 predict,
                 nb_iters,
                 eps_max,
                 step_size,
                 resolution):
        self.pixel_model = PixelModel(predict)
        self.fog_obj = FogAttackBase(
            nb_its=nb_iters,
            eps_max=eps_max,
            step_size=step_size,
            resol=resolution)
        print("finish init fogattack")

    def perturb(self, images, labels):
        pixel_img = inverse_transform(images.clamp(-1., 1.)).detach().clone()
        pixel_ret = self.fog_obj._forward(
            pixel_model=self.pixel_model,
            pixel_img=pixel_img,
            target=labels)

        return transform(pixel_ret)
