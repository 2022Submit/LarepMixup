# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""


#---train
from logging import error, exception
import os
import json
import tempfile
import torch
from torch import tensor
from torchvision.models.shufflenetv2 import channel_shuffle
import utils.stylegan2ada.dnnlib as dnnlib       
from utils.stylegan2ada.training import training_loop
from utils.stylegan2ada.metrics import metric_main
from utils.stylegan2ada.torch_utils import training_stats
from utils.stylegan2ada.torch_utils import custom_ops

#----project
import copy
from time import perf_counter
import imageio
import numpy as np
import PIL.Image
from numpy.core.records import array
import torch.nn.functional as F
import utils.stylegan2ada.legacy as legacy
import utils.sampler
import re
from typing import List, Optional
import click
from torchvision.transforms import transforms


class MaggieStylegan2ada:

    def __init__(self, args):
        #   initialize the parameters
        self._args = args
        # self._snapshot_network_pkls = None  # snapshot_network_pkls list

    def snapshot_network_pkls(self):        
        return self._snapshot_network_pkls

    def train(self,exp_result_dir, stylegan2ada_config_kwargs):

        self._exp_result_dir = exp_result_dir
        self._stylegan2ada_config_kwargs = stylegan2ada_config_kwargs
        self.__train__()

    def __train__(self):
        snapshot_network_pkls = self.__trainmain__(self._args, self._exp_result_dir, **self._stylegan2ada_config_kwargs)
        self._snapshot_network_pkls = snapshot_network_pkls

    def __trainmain__(self, opt, exp_result_dir, **config_kwargs):                                        
        print("running stylegan2ada train main()...............")
        dry_run = opt.dry_run   
        dnnlib.util.Logger(should_flush=True)
        run_desc, args = self.__setup_training_loop_kwargs__(**config_kwargs)         
        
        args.run_dir = os.path.join(exp_result_dir, f'{run_desc}')         
        assert not os.path.exists(args.run_dir)             

        # Print options.
        print()
        print('Training options:')
        print(json.dumps(args, indent=2))
        print()
        print(f'Output directory:   {args.run_dir}')
        print(f'Training data:      {args.training_set_kwargs.path}')
        print(f'Training duration:  {args.total_kimg} kimg')
        print(f'Number of GPUs:     {args.num_gpus}')
        print(f'Number of images:   {args.training_set_kwargs.max_size}')
        print(f'Image resolution:   {args.training_set_kwargs.resolution}')
        print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
        print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
        print()

        # Dry run?
        if dry_run:
            print('Dry run; exiting.')
            return

        # Create output directory.
        print('Creating output directory...')
        os.makedirs(args.run_dir)
        with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(args, f, indent=2)

        # Launch processes.
        print('Launching processes...')
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:
                snapshot_network_pkls = self.__subprocess_fn__(rank=0, args=args, temp_dir=temp_dir)            
            else:
                snapshot_network_pkls = torch.multiprocessing.spawn(fn=self.__subprocess_fn__, args=(args, temp_dir), nprocs=args.num_gpus)
        return snapshot_network_pkls
    
    def __setup_training_loop_kwargs__(self,
        # General options (not included in desc).
        gpus       = None, # Number of GPUs: <int>, default = 1 gpu
        snap       = None, # Snapshot interval: <int>, default = 50 ticks
        metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
        seed       = None, # Random seed: <int>, default = 0

        # Dataset.
        data       = None, # Training dataset (required): <path>
        cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
        subset     = None, # Train with only N images: <int>, default = all
        mirror     = None, # Augment dataset with x-flips: <bool>, default = False

        # Base config.
        cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
        gamma      = None, # Override R1 gamma: <float>
        kimg       = None, # Override training duration: <int>
        # batch      = None, # Override batch size: <int>
        batch_size = None, # Override batch size: <int>

        # Discriminator augmentation.
        aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
        p          = None, # Specify p for 'fixed' (required): <float>
        target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
        augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

        # Transfer learning.
        resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
        freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

        # Performance options (not included in desc).
        fp32       = None, # Disable mixed-precision training: <bool>, default = False
        nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
        allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
        nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
        workers    = None, # Override number of DataLoader workers: <int>, default = 3
        pretrain_pkl_path =None, # pretrained stylegan2ADA model pkl path
    ):
        args = dnnlib.EasyDict()
        # ------------------------------------------
        # General options: gpus, snap, metrics, seed
        # ------------------------------------------

        if gpus is None:
            gpus = 1
        assert isinstance(gpus, int)
        if not (gpus >= 1 and gpus & (gpus - 1) == 0):
            raise Exception('--gpus must be a power of two')                                                                   
        args.num_gpus = gpus

        if snap is None:
            snap = 50
        assert isinstance(snap, int)
        if snap < 1:
            raise Exception('--snap must be at least 1')                                                           
        args.image_snapshot_ticks = snap
        args.network_snapshot_ticks = snap

        if metrics is None:
            metrics = ['fid50k_full']                                                                                          
        assert isinstance(metrics, list)
        if not all(metric_main.is_valid_metric(metric) for metric in metrics):
            raise Exception('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
        args.metrics = metrics

        if seed is None:
            seed = 0
        assert isinstance(seed, int)
        args.random_seed = seed      

        # -----------------------------------
        # Dataset: data, cond, subset, mirror
        # -----------------------------------

        assert data is not None
        assert isinstance(data, str)
        args.training_set_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)  

        args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
        try:
            training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs)                                      #   subclass of training.dataset.Dataset   
            args.training_set_kwargs.resolution = training_set.resolution                                                       #   be explicit about resolution                            
            args.training_set_kwargs.use_labels = training_set.has_labels                                                       #   be explicit about labels                               
            args.training_set_kwargs.max_size = len(training_set)                                                               #   be explicit about dataset size                                  
            desc = training_set.name                                                                                            #   desc=cifar10                                              
            del training_set # conserve memory                                                                                  #   
        except IOError as err:
            raise Exception(f'--data: {err}')

        if cond is None:
            cond = False
        assert isinstance(cond, bool)
        if cond:                                                                                                            
            if not args.training_set_kwargs.use_labels:
                raise Exception('--cond=True requires labels specified in dataset.json')
            desc += '-cond'                                                                                                    
        else:                                                                                                                  
            args.training_set_kwargs.use_labels = False

        if subset is not None:
            assert isinstance(subset, int)
            if not 1 <= subset <= args.training_set_kwargs.max_size:                                                  
                raise Exception(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
            desc += f'-subset{subset}'                                                                                        
            if subset < args.training_set_kwargs.max_size:                                                                  
                args.training_set_kwargs.max_size = subset                                                                  
                args.training_set_kwargs.random_seed = args.random_seed

        if mirror is None:
            mirror = False
        assert isinstance(mirror, bool)
        if mirror:                                                                                                         
            desc += '-mirror'                                                                                                 
            args.training_set_kwargs.xflip = True                                                                       

        # ------------------------------------
        # Base config: cfg, gamma, kimg, batch
        # ------------------------------------

        if cfg is None:
            cfg = 'auto'                                                                                                      
        assert isinstance(cfg, str)
        desc += f'-{cfg}'                                                                                                   

        cfg_specs = {
            'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
            'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
            'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
            'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
            'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
            'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        }

        assert cfg in cfg_specs
        spec = dnnlib.EasyDict(cfg_specs[cfg])
        if cfg == 'auto':
            desc += f'{gpus:d}'                                                                                                 #   desc=cifar10-subset-auto1
            spec.ref_gpus = gpus
            res = args.training_set_kwargs.resolution
            spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus)                                                         
            spec.mbstd = min(spec.mb // gpus, 4)                                                                               
            spec.fmaps = 1 if res >= 512 else 0.5                                                                         
            spec.lrate = 0.002 if res >= 1024 else 0.0025
            spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
            spec.ema = spec.mb * 10 / 32

        args.G_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())           
        args.D_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())   

        args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)  
        args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
        args.G_kwargs.mapping_kwargs.num_layers = spec.map
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4                                           
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256                                            
        args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd

        args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)            
        args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        args.loss_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.loss.StyleGAN2Loss', r1_gamma=spec.gamma)

        args.total_kimg = spec.kimg
        args.batch_size = spec.mb
        args.batch_gpu = spec.mb // spec.ref_gpus
        args.ema_kimg = spec.ema
        args.ema_rampup = spec.ramp

        if cfg == 'cifar':
            args.loss_kwargs.pl_weight = 0                                                                                      
            args.loss_kwargs.style_mixing_prob = 0                                                                              
            args.D_kwargs.architecture = 'orig'                                                                                 
        if gamma is not None:
            assert isinstance(gamma, float)
            if not gamma >= 0:
                raise Exception('--gamma must be non-negative')                                                                 
            desc += f'-gamma{gamma:g}'                                                                                      
            args.loss_kwargs.r1_gamma = gamma

        if kimg is not None:
            assert isinstance(kimg, int)
            if not kimg >= 1:
                raise Exception('--kimg must be at least 1')
            desc += f'-kimg{kimg:d}'
            args.total_kimg = kimg

        if batch_size is not None:
            assert isinstance(batch_size, int)
            if not (batch_size >= 1 and batch_size % gpus == 0):
                raise Exception('--batch_size must be at least 1 and divisible by --gpus')
            desc += f'-batch{batch_size}'                                                                                     
            args.batch_size = batch_size
            args.batch_gpu = batch_size // gpus

        # ---------------------------------------------------
        # Discriminator augmentation: aug, p, target, augpipe                                                                  
        # ---------------------------------------------------

        if aug is None:
            aug = 'ada'
        else:
            assert isinstance(aug, str)
            desc += f'-{aug}'                                                                                                   #   desc=cifar10-subset-auto1-batch32-ada

        if aug == 'ada':
            args.ada_target = 0.6

        elif aug == 'noaug':
            pass

        elif aug == 'fixed':
            if p is None:
                raise Exception(f'--aug={aug} requires specifying --p')

        else:
            raise Exception(f'--aug={aug} not supported')

        if p is not None:
            assert isinstance(p, float)
            if aug != 'fixed':
                raise Exception('--p can only be specified with --aug=fixed')
            if not 0 <= p <= 1:
                raise Exception('--p must be between 0 and 1')
            desc += f'-p{p:g}'
            args.augment_p = p

        if target is not None:
            assert isinstance(target, float)
            if aug != 'ada':
                raise Exception('--target can only be specified with --aug=ada')
            if not 0 <= target <= 1:
                raise Exception('--target must be between 0 and 1')
            desc += f'-target{target:g}'
            args.ada_target = target

        assert augpipe is None or isinstance(augpipe, str)
        if augpipe is None:
            augpipe = 'bgc'
        else:
            if aug == 'noaug':
                raise Exception('--augpipe cannot be specified with --aug=noaug')
            desc += f'-{augpipe}'                                                                                               #   desc=cifar10-subset-auto1-batch32-ada-bgc

        augpipe_specs = {
            'blit':   dict(xflip=1, rotate90=1, xint=1),
            'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
            'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'filter': dict(imgfilter=1),
            'noise':  dict(noise=1),
            'cutout': dict(cutout=1),
            'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
            'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
            'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
            'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
            'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
        }

        assert augpipe in augpipe_specs
        if aug != 'noaug':
            # args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])
            args.augment_kwargs = dnnlib.EasyDict(class_name='utils.stylegan2ada.training.augment.AugmentPipe', **augpipe_specs[augpipe])

        # ----------------------------------
        # Transfer learning: resume, freezed
        # ----------------------------------

        resume_specs = {
            'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
            'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
            'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
            'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
            'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
        }

        assert resume is None or isinstance(resume, str)
        if resume is None:
            resume = 'noresume'
        elif resume == 'noresume':
            desc += '-noresume'                                                                                                 #   desc=cifar10-subset-auto1-batch32-ada-bgc-noresume                      
        elif resume in resume_specs:
            desc += f'-resume{resume}'
            args.resume_pkl = resume_specs[resume]                                                                              #   predefined url
        else:
            desc += '-resumecustom'
            args.resume_pkl = resume                                                                                            #   custom path or url

        # maggie-----------
        if pretrain_pkl_path is not None:
            args.resume_pkl = pretrain_pkl_path  
            print("args.resume_pkl:",args.resume_pkl)
        #---------------------
        
        if resume != 'noresume':
            args.ada_kimg = 100                                                                                                 #   make ADA react faster at the beginning
            args.ema_rampup = None                                                                                              #   disable EMA rampup

        if freezed is not None:
            assert isinstance(freezed, int)
            if not freezed >= 0:
                raise Exception('--freezed must be non-negative')
            desc += f'-freezed{freezed:d}'
            args.D_kwargs.block_kwargs.freeze_layers = freezed

        # -------------------------------------------------
        # Performance options: fp32, nhwc, nobench, workers
        # -------------------------------------------------

        if fp32 is None:
            fp32 = False
        assert isinstance(fp32, bool)
        if fp32:
            args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
            args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

        if nhwc is None:
            nhwc = False
        assert isinstance(nhwc, bool)
        if nhwc:
            args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

        if nobench is None:
            nobench = False
        assert isinstance(nobench, bool)
        if nobench:
            args.cudnn_benchmark = False

        if allow_tf32 is None:
            allow_tf32 = False
        assert isinstance(allow_tf32, bool)
        if allow_tf32:
            args.allow_tf32 = True

        if workers is not None:
            assert isinstance(workers, int)
            if not workers >= 1:
                raise Exception('--workers must be at least 1')
            args.data_loader_kwargs.num_workers = workers

        return desc, args                                                                                                 
    
    def __subprocess_fn__(self,rank, args, temp_dir):
        dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

        # Init torch.distributed.
        if args.num_gpus > 1:
            init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
            if os.name == 'nt':
                init_method = 'file:///' + init_file.replace('\\', '/')
                torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
            else:
                init_method = f'file://{init_file}'
                torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

        # Init torch_utils.
        sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
        training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
        if rank != 0:
            custom_ops.verbosity = 'none'

     
        snapshot_network_pkls = training_loop.training_loop(rank=rank, **args)                                                 
    
        return snapshot_network_pkls

    def wyset(self):
        return self.projected_w_set,self.projected_y_set

    def project(self,exp_result_dir, ori_x_set = None, ori_y_set = None,batch_index=None):
        self._exp_result_dir = exp_result_dir
        self._batch_index = batch_index
        projected_w_set, projected_y_set = self.__projectmain__(self._args, self._exp_result_dir,ori_x_set, ori_y_set)
        self.projected_w_set = projected_w_set
        self.projected_y_set = projected_y_set

    def __projectmain__(self, opt, exp_result_dir,ori_x_set, ori_y_set):
        print("running projecting main()..............")

        if ori_x_set is not None :   
            self.ori_x_set = ori_x_set
            self.ori_y_set = ori_y_set
            print("Project original images from images tensor set !")
            projected_w_set, projected_y_set = self.__ramxyproject__()

        else:
            print("Project original images from view dataset path !")
            if opt.target_fname == None:
                print(f'Project samples of the *{self._args.dataset}* dataset !')
                projected_w_set, projected_y_set = self.__run_projection_dataset_fromviewfolder(opt,exp_result_dir)

            elif opt.target_fname != None:
                print('Project single sample of the *{self._args.dataset}* dataset')
                projected_w_set, projected_y_set = self.__run_projection__(
                    network_pkl = opt.gen_network_pkl,
                    target_fname = opt.target_fname,
                    outdir = exp_result_dir,
                    save_video = opt.save_video,
                    seed = opt.seed,
                    num_steps = opt.num_steps,    
                    image_name = 'test'
                )

        return projected_w_set, projected_y_set         

    def __labelnames__(self):
        opt = self._args
        # print("opt.dataset:",opt.dataset)
        
        label_names = []
        
        if opt.dataset == 'cifar10':
            label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

        elif opt.dataset == 'cifar100': # = cle_train_dataloader.dataset.classes
            label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        
        elif opt.dataset =='svhn':
            label_names = ['0','1','2','3','4','5','6','7','8','9']

        elif opt.dataset =='kmnist':
            label_names = ['0','1','2','3','4','5','6','7','8','9']
        
        elif opt.dataset =='stl10': # cle_train_dataloader.dataset.classes 标签序号是0-9, dataloader 已调整数字0的标签为0
            label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        
        elif opt.dataset =='imagenetmixed10':
            label_names = ['dog,','bird','insect','monkey','car','feline','truck','fruit','fungus','boat']        
        else:
            raise error            
        
        return label_names

    def __ramxyproject__(self):
        opt = self._args
        exp_result_dir = self._exp_result_dir
        exp_result_dir = os.path.join(exp_result_dir,f'project-{opt.dataset}-trainset')    

        os.makedirs(exp_result_dir,exist_ok=True)    

        target_x_set = self.ori_x_set
        target_y_set = self.ori_y_set

        projected_x_set = []
        projected_y_set = []

        for index in range(len(target_x_set)):                                                                         
            
            if  self._args.project_target_num != None:
                if index + opt.batch_size * self._batch_index < self._args.project_target_num:                                                                          
                    projected_w, projected_y = self.__xyproject__(
                        network_pkl = opt.gen_network_pkl,
                        target_pil = target_x_set[index],
                        outdir = exp_result_dir,
                        save_video = opt.save_video,
                        seed = opt.seed,
                        num_steps = opt.num_steps,
                        projected_img_index = index + opt.batch_size * self._batch_index,
                        laber_index = target_y_set[index]
                    )
                    projected_x_set.append(projected_w)
                    projected_y_set.append(projected_y)         
            else:
                print("projecting samples num:",len(target_x_set))
                projected_w, projected_y = self.__xyproject__(
                    network_pkl = opt.gen_network_pkl,
                    target_pil = target_x_set[index],
                    outdir = exp_result_dir,
                    save_video = opt.save_video,
                    seed = opt.seed,
                    num_steps = opt.num_steps,
                    projected_img_index = index + opt.batch_size * self._batch_index,
                    laber_index = target_y_set[index]
                )
                projected_x_set.append(projected_w)
                projected_y_set.append(projected_y)         
                     
        print('Finished dataset projecting !')
        return projected_x_set,projected_y_set         

    def __xyproject__(self,                                            
        network_pkl: str,              
        target_pil,                    
        outdir: str,                   
        save_video: bool,               
        seed: int,                
        num_steps: int,                 
        projected_img_index:int,      
        laber_index: int
    ):

        print(f"projecting {projected_img_index:08d} image:")

        np.random.seed(seed)                                                                                                    
        torch.manual_seed(seed)

        # Load networks.
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)                         
        
        if self._args.dataset =='cifar10' or self._args.dataset =='cifar100' or self._args.dataset =='svhn' or self._args.dataset =='stl10' or self._args.dataset =='imagenetmixed10':
            if self._args.dataset =='svhn' or self._args.dataset =='stl10' or self._args.dataset =='imagenetmixed10':
                target_pil = target_pil.transpose([1, 2, 0])

            target_pil = PIL.Image.fromarray(target_pil, 'RGB')  
        
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)

            target_uint8 = np.array(target_pil, dtype=np.uint8)
            target_uint8 = target_uint8.transpose([2, 0, 1])

        elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
            target_pil = target_pil.numpy()                         
            target_pil = PIL.Image.fromarray(target_pil, 'L')     
            
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS) 

            target_uint8 = np.array(target_pil, dtype=np.uint8)                                
            target_uint8 = torch.tensor(target_uint8).unsqueeze(0)
            target_uint8 = target_uint8.numpy()


        projected_w_steps = self.__project__(
            G,
            target=torch.tensor(target_uint8, device=device),                                              
            num_steps=num_steps,
            device=device,
            verbose=True
        )        

        os.makedirs(outdir, exist_ok=True)

        classification = self.__labelnames__() 

        label_name = classification[int(laber_index)]
        print(f"label = {laber_index:04d}-{classification[int(laber_index)]}")

        projected_w = projected_w_steps[-1]                                                
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')           
        
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
            synth_image = PIL.Image.fromarray(synth_image, 'RGB')
        elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
            synth_image = synth_image.transpose([2, 0, 1])
            synth_image = synth_image[0]
            synth_image = PIL.Image.fromarray(synth_image, 'L')

        np.savez(f'{outdir}/{projected_img_index:08d}-{int(laber_index)}-{label_name}-projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())      

        projected_w_y = int(laber_index) * torch.ones(projected_w.size(0), dtype = int) 
        np.savez(f'{outdir}/{projected_img_index:08d}-{int(laber_index)}-{label_name}-label.npz', w = projected_w_y.unsqueeze(0).cpu().numpy())      

        projected_w = projected_w                                                                           
        projected_y = int(laber_index)                                                                                              
        projected_y = projected_y * torch.ones(G.mapping.num_ws, dtype = int)                               
        return projected_w,projected_y
        

    def __run_projection_dataset_fromviewfolder(self,opt,exp_result_dir):

        peojected_w_set = []
        projected_y_set = []

        exp_result_dir = os.path.join(exp_result_dir,f'project-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)    
        
        file_dir=os.listdir(opt.viewdataset_path)
        file_dir.sort()
        filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.png']                                        

        for index, filename in enumerate(filenames):                                                                       
            if  self._args.project_target_num != None:
                if index < self._args.project_target_num:                                                                         
                    print(f"projecting {self._args.project_target_num} cle samples !  ")
                    img_name = filename[:-4]
                    img_index = img_name[0:8]
                    label_number = img_name[9:10]
                    label = img_name[11:]
                    target_fname = os.path.join(opt.viewdataset_path,filename)                                                
                    projected_w,projected_y = self.__run_projection__(
                        network_pkl = opt.gen_network_pkl,
                        target_fname = target_fname,
                        outdir = exp_result_dir,
                        save_video = opt.save_video,
                        seed = opt.seed,
                        num_steps = opt.num_steps,
                        image_name = img_name,
                        projected_img_index = index
                                
                    )

                    peojected_w_set.append(projected_w)
                    projected_y_set.append(projected_y)         
            else:
                print(f"projecting the whole {len(filenames)} cle samples !  ")

                img_name = filename[:-4]
                img_index = img_name[0:8]
                label_number = img_name[9:10]
                label = img_name[11:]
                target_fname = os.path.join(opt.viewdataset_path,filename)                                               {filename}'   

                projected_w,projected_y = self.__run_projection__(
                    network_pkl = opt.gen_network_pkl,
                    target_fname = target_fname,
                    outdir = exp_result_dir,
                    save_video = opt.save_video,
                    seed = opt.seed,
                    num_steps = opt.num_steps,
                    image_name = img_name,
                    projected_img_index = index
                            
                )

                peojected_w_set.append(projected_w)
                projected_y_set.append(projected_y)         
                                   
        print('Finished dataset projecting !')
        return peojected_w_set, projected_y_set

    def __run_projection__(self,
        network_pkl: str,             
        target_fname: str,             
        outdir: str,                  
        save_video: bool,             
        seed: int,                     
        num_steps: int,                 
        image_name: str,                
        projected_img_index:int        
    ):

        print(f"projecting {projected_img_index:08d} image:")
        
        np.random.seed(seed)                                                                                            
        torch.manual_seed(seed)

        # Load networks.
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)                                           
        # Load target image.
        target_pil = PIL.Image.open(target_fname).convert('RGB')
        # print(target_pil)
        
        w, h = target_pil.size
        # print('target_pil.size=%s' % target_pil)

        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)

        target_uint8 = np.array(target_pil, dtype=np.uint8)

        start_time = perf_counter()
        projected_w_steps = self.__project__(
            G,
            target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),                                              #   pylint: disable=not-callable
            num_steps=num_steps,
            device=device,
            verbose=True
        )

        os.makedirs(outdir, exist_ok=True)
        if save_video:                                                                                                          
            video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')                 
            print (f'Saving optimization progress video "{outdir}/proj.mp4"')
            for projected_w in projected_w_steps:
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')                                         #   projected_w.unsqueeze(0).shape: torch.size[1,8,512]
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
            video.close()

        img_index = image_name[0:8]
        label_number = image_name[9:10]                                                                                        
        label_number = int(label_number)                                                                                        #   str -> int
        label = image_name[11:]
        
        target_pil.save(f'{outdir}/original-{img_index}-{label_number}-{label}.png')                                          
        
        projected_w = projected_w_steps[-1]                                                                                     
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')                                  
        
        synth_image = (synth_image + 1) * (255/2)
        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        synth_image = PIL.Image.fromarray(synth_image, 'RGB')
        synth_image.save(f'{outdir}/projected-{img_index}-{label_number}-{label}.png')
        np.savez(f'{outdir}/projected_w-{img_index}-{label_number}-{label}.npz', w=projected_w.unsqueeze(0).cpu().numpy())     
        projected_w = projected_w                                                                                               
        projected_y = label_number                                                                                      
        projected_y = projected_y * torch.ones(G.mapping.num_ws, dtype = int)                                             
        print("projected_y: ",projected_y)                                                                                  
        return projected_w,projected_y

    def __project__(self,
        G,
        target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps                  = 1000,
        w_avg_samples              = 10000,
        initial_learning_rate      = 0.1,
        initial_noise_factor       = 0.05,
        lr_rampdown_length         = 0.25,
        lr_rampup_length           = 0.05,
        noise_ramp_length          = 0.75,
        regularize_noise_weight    = 1e5,
        verbose                    = False,
        device: torch.device
    ):

        assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

        def logprint(*args):
            if verbose:
                print(*args)

        G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'      
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(device)

        target_images = target.unsqueeze(0).to(device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        
        if self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
            target_images = target_images.expand(-1, 3, -1, -1).clone() 
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
        w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(num_steps):
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            synth_images = G.synthesis(ws, noise_mode='const')

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255/2)
            if synth_images.shape[2] > 256:
                synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

            if self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
                synth_images = synth_images.expand(-1, 3, -1, -1).clone()
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

            dist = (target_features - synth_features).square().sum()
            # print("dist=",dist)
            # raise error
            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            w_out[step] = w_opt.detach()[0]             
            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()
        return w_out.repeat([1, G.mapping.num_ws, 1])
    
    def mixwyset(self):
        return self.interpolated_w_set, self.interpolated_y_set

    def interpolate(self, exp_result_dir, projected_w_set = None, projected_y_set = None):
        self._exp_result_dir = exp_result_dir
        interpolated_w_set, interpolated_y_set = self.__interpolatemain__(self._args, self._exp_result_dir, projected_w_set, projected_y_set)
        self.interpolated_w_set = interpolated_w_set
        self.interpolated_y_set = interpolated_y_set

    def __interpolatemain__(self, opt, exp_result_dir, projected_w_set, projected_y_set):
        # print("running interpolate main()..............")

        if projected_w_set is not None :     
            self.projected_w_set = projected_w_set
            self.projected_y_set = projected_y_set
            # print("Interpolate projectors from projectors numpy ndarray !")
            interpolated_w_set, interpolated_y_set = self.__ramwymixup__()        
        else:
            print("Interpolate projectors from projectors npz files !")
            if opt.projected_dataset != None:               
                interpolated_w_set, interpolated_y_set = self.__DatasetMixup__(opt,exp_result_dir) 
            elif opt.projected_dataset == None:  
                print("opt.projected_w3:",opt.projected_w3)
                if opt.projected_w3 == None:
                    interpolated_w_set, interpolated_y_set = self.__TwoMixup__(opt,exp_result_dir)
                elif opt.projected_w3 != None:       
                    interpolated_w_set, interpolated_y_set = self.__ThreeMixup__(opt,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

    def __ramwymixup__(self):
        opt = self._args
        exp_result_dir = self._exp_result_dir
        device = torch.device('cuda')        
        if self._args.defense_mode == 'rmt':
            projected_w_set_x = self.projected_w_set                                                       #   numpy -> tensor
            projected_w_set_y = self.projected_y_set               
            projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float()           #   CPU tensor 

            interpolated_w_set, interpolated_y_set = self.__getmixedbatchwy__(opt, projected_w_set_x, projected_w_set_y)

        else:
            projected_w_set_x = torch.tensor(self.projected_w_set).to(device)                                                      
            projected_w_set_y = torch.tensor(self.projected_y_set).to(device)                                                         

            projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                   
            interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)
            
        return interpolated_w_set, interpolated_y_set   

    def __getmixedbatchwy__(self, opt, projected_w_set_x, projected_w_set_y):
        repeat_num = projected_w_set_x.size(1)

        if opt.mix_w_num == 2:
            batch_size = projected_w_set_x.size()[0]

            shuffle_index = torch.randperm(batch_size)

            shuffled_projected_w_set_x = projected_w_set_x[shuffle_index,:]
            shuffled_projected_w_set_y = projected_w_set_y[shuffle_index,:]
            
            projected_w_set_x = projected_w_set_x[:, 0, :].squeeze(1)                           #   [4,8,512] --> [4,1,512] --> [4,512]
            shuffled_projected_w_set_x = shuffled_projected_w_set_x[:, 0, :].squeeze(1)         #   [4,8,512] --> [4,1,512] --> [4,512]

            projected_w_set_y = projected_w_set_y[:, 0, :].squeeze(1)                           #   [4,8,10] --> [4,1,10] --> [4,10]
            shuffled_projected_w_set_y = shuffled_projected_w_set_y[:, 0, :].squeeze(1)

            if opt.mix_mode == 'basemixup':
                interpolated_w_set, interpolated_y_set = self.__BaseMixup2__(projected_w_set_x, shuffled_projected_w_set_x, opt.sample_mode, projected_w_set_y, shuffled_projected_w_set_y)
            elif opt.mix_mode == 'maskmixup':
                interpolated_w_set, interpolated_y_set = self.__MaskMixup2__(projected_w_set_x, shuffled_projected_w_set_x, opt.sample_mode, projected_w_set_y, shuffled_projected_w_set_y)
            else:
                raise Exception('please input valid mix_mode')
            
            interpolated_w_set = interpolated_w_set.unsqueeze(1)
            interpolated_y_set = interpolated_y_set.unsqueeze(1)

            interpolated_w_set = interpolated_w_set.expand(interpolated_w_set.size()[0],repeat_num,interpolated_w_set.size()[2])
            interpolated_y_set = interpolated_y_set.expand(interpolated_y_set.size()[0],repeat_num,interpolated_y_set.size()[2])

        elif opt.mix_w_num==3:
            # print("ternary mixup")     
            batch_size = projected_w_set_x.size()[0]

            shuffle_index_a = torch.randperm(batch_size)
            shuffled_projected_w_set_x_a = projected_w_set_x[shuffle_index_a,:]
            shuffled_projected_w_set_y_a = projected_w_set_y[shuffle_index_a,:]

            shuffle_index_b = torch.randperm(batch_size)
            shuffled_projected_w_set_x_b = projected_w_set_x[shuffle_index_b,:]
            shuffled_projected_w_set_y_b = projected_w_set_y[shuffle_index_b,:]

            projected_w_set_x = projected_w_set_x[:, 0, :].squeeze(1)                           #   [4,8,512] --> [4,1,512] --> [4,512]
            projected_w_set_y = projected_w_set_y[:, 0, :].squeeze(1)                           #   [4,8,10] --> [4,1,10] --> [4,10]

            shuffled_projected_w_set_x_a = shuffled_projected_w_set_x_a[:, 0, :].squeeze(1)     #   [4,8,512] --> [4,1,512] --> [4,512]
            shuffled_projected_w_set_y_a = shuffled_projected_w_set_y_a[:, 0, :].squeeze(1)     #   [4,8,10] --> [4,1,10] --> [4,10]       

            shuffled_projected_w_set_x_b = shuffled_projected_w_set_x_b[:, 0, :].squeeze(1)     #   [4,8,512] --> [4,1,512] --> [4,512]
            shuffled_projected_w_set_y_b = shuffled_projected_w_set_y_b[:, 0, :].squeeze(1)     #   [4,8,10] --> [4,1,10] --> [4,10]   


            if opt.mix_mode == 'basemixup':
                interpolated_w_set, interpolated_y_set = self.__BaseMixup3__(projected_w_set_x, shuffled_projected_w_set_x_a, shuffled_projected_w_set_x_b, opt.sample_mode, projected_w_set_y, shuffled_projected_w_set_y_a,shuffled_projected_w_set_y_b)

            elif opt.mix_mode == 'maskmixup':
                interpolated_w_set, interpolated_y_set = self.__MaskMixup3__(projected_w_set_x, shuffled_projected_w_set_x_a, shuffled_projected_w_set_x_b, opt.sample_mode, projected_w_set_y, shuffled_projected_w_set_y_a,shuffled_projected_w_set_y_b)

            else:
                raise Exception('please input valid mix_mode')

            interpolated_w_set = interpolated_w_set.unsqueeze(1)
            interpolated_y_set = interpolated_y_set.unsqueeze(1)

            interpolated_w_set = interpolated_w_set.expand(interpolated_w_set.size()[0],repeat_num,interpolated_w_set.size()[2])
            interpolated_y_set = interpolated_y_set.expand(interpolated_y_set.size()[0],repeat_num,interpolated_y_set.size()[2])


        return interpolated_w_set, interpolated_y_set

    def __getmixededwy__(self,opt, projected_w_set_x,projected_w_set_y,exp_result_dir):

        exp_result_dir = os.path.join(exp_result_dir,f'interpolate-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)    

        classification = self.__labelnames__() 
        print("classification label name:",classification)

        interpolated_w_set = []
        interpolated_y_set = []
        print("projected_w_set_x.shape:",projected_w_set_x.shape)           #   projected_w_set_x.shape: torch.Size([38, 10, 512])
        print("projected_w_set_y.shape:",projected_w_set_y.shape)
        
        mix_num = 0
        if opt.mix_w_num == 2:      
            print("--------------------Dual mixup----------------------")
            for i in range(len(projected_w_set_x)):                      

                if i in [2,3,26,27,28,29]:  #6

                    for j in range(len(projected_w_set_x)):
                        if j in [0,1,2,3,20,21,24,25,26,27,28,29]: #12
                        # if j in [8,34]:

                            if j != i:
                                if mix_num < opt.mix_img_num:       
                                    w1 = projected_w_set_x[i][-1].unsqueeze(0)
                                    y1 = projected_w_set_y[i][-1].unsqueeze(0)
               
                                    w2 = projected_w_set_x[j][-1].unsqueeze(0)
                                    y2 = projected_w_set_y[j][-1].unsqueeze(0) 

                                    _, w1_label_index = torch.max(y1, 1)    
                                    _, w2_label_index = torch.max(y2, 1)  

                                    w1_label_name = f"{classification[int(w1_label_index)]}"
                                    w2_label_name = f"{classification[int(w2_label_index)]}"

                                    if w1_label_name == w2_label_name:
                                        print("mixup same class")
                                    else:
                                        print("mixup different classes")

                                    print("w1_label_name:",w1_label_name)
                                    print("w2_label_name:",w2_label_name)
                                    

                                    if opt.mix_mode == 'basemixup':
                                        w_mixed, y_mixed = self.__BaseMixup2__(w1,w2,opt.sample_mode,y1,y2)
                                    elif opt.mix_mode == 'maskmixup':
                                        w_mixed, y_mixed = self.__MaskMixup2__(w1,w2,opt.sample_mode,y1,y2)
                                    elif opt.mix_mode == 'adversarialmixup':
                                        w_mixed = self.__AdversarialMixup2__(w1,w2,opt.sample_mode)
                                    else:
                                        raise Exception('please input valid mix_mode')
                                
                                    repeat_num = projected_w_set_x.size(1)
                                    # w_mixed = w_mixed.repeat([8,1])       
                                    # y_mixed = y_mixed.repeat([8,1])
                                    w_mixed = w_mixed.repeat([repeat_num,1])       
                                    y_mixed = y_mixed.repeat([repeat_num,1])                    

                                    np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}-mixed_projected_w.npz', w=w_mixed.unsqueeze(0).cpu().numpy())                  
                                    np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}-mixed_label.npz', w = y_mixed.unsqueeze(0).cpu().numpy())               

                                    interpolated_w_set.append(w_mixed)
                                    interpolated_y_set.append(y_mixed)

                                    mix_num = mix_num + 1

        elif opt.mix_w_num == 3:
            print("-------------------Ternary mixup----------------------")
            for i in range(len(projected_w_set_x)):
                if i in [2,3,26,27]:  #4

                    for j in range(len(projected_w_set_x)):
                        if j in [2,3,26,27,28,29]: #6

                            for k in range(len(projected_w_set_x)):
                                if k in [0,1,2,3,20,21,24,25,26,27,28,29]:   #12
                                    if k != j and j != i :
                                        if mix_num < opt.mix_img_num:
                                            w1 = projected_w_set_x[i][-1].unsqueeze(0)
                                            y1 = projected_w_set_y[i][-1].unsqueeze(0)
                                           
                                            w2 = projected_w_set_x[j][-1].unsqueeze(0)
                                            y2 = projected_w_set_y[j][-1].unsqueeze(0) 
              
                                            w3 = projected_w_set_x[k][-1].unsqueeze(0)
                                            y3 = projected_w_set_y[k][-1].unsqueeze(0) 
                           

                                            _, w1_label_index = torch.max(y1, 1)    
                                            _, w2_label_index = torch.max(y2, 1)  
                                            _, w3_label_index = torch.max(y3, 1)    
                                            w1_label_name = f"{classification[int(w1_label_index)]}"
                                            w2_label_name = f"{classification[int(w2_label_index)]}"
                                            w3_label_name = f"{classification[int(w3_label_index)]}"
                            
                                            if w1_label_name == w2_label_name and w2_label_name == w3_label_name:
                                                print("mixup same class")

                                            else:
                                                print("mixup different classes")

                                            print("w1_label_name:",w1_label_name)
                                            print("w2_label_name:",w2_label_name)
                                            print("w3_label_name:",w2_label_name)
                                            
                                            # print("opt.mix_mode:",opt.mix_mode)
                                            if opt.mix_mode == 'basemixup':
                                                w_mixed, y_mixed = self.__BaseMixup3__(w1,w2,w3,opt.sample_mode,y1,y2,y3)
                                            elif opt.mix_mode == 'maskmixup':
                                                w_mixed, y_mixed = self.__MaskMixup3__(w1,w2,w3,opt.sample_mode,y1,y2,y3)
                                            else:
                                                raise Exception('please input valid mix_mode')
                                        
                                            repeat_num = projected_w_set_x.size(1)
                                            w_mixed = w_mixed.repeat([repeat_num,1])       
                                            y_mixed = y_mixed.repeat([repeat_num,1]) 

                                            np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}+{k:08d}-{int(w3_label_index)}-{w3_label_name}-mixed_projected_w.npz', w=w_mixed.unsqueeze(0).cpu().numpy())  
                                            np.savez(f'{exp_result_dir}/{i:08d}-{int(w1_label_index)}-{w1_label_name}+{j:08d}-{int(w2_label_index)}-{w2_label_name}+{k:08d}-{int(w3_label_index)}-{w3_label_name}-mixed_label.npz', w = y_mixed.unsqueeze(0).cpu().numpy())      
                                            interpolated_w_set.append(w_mixed)
                                            interpolated_y_set.append(y_mixed)           

                                            mix_num = mix_num + 1                       

        return interpolated_w_set, interpolated_y_set

    def __BaseMixup2__(self,w1,w2,sample_mode,y1,y2):                                                                    


        is_2d = True if len(w1.size()) == 2 else False                                                                   
        if sample_mode == 'uniformsampler':
            alpha = utils.sampler.UniformSampler(w1.size(0), w1.size(1), is_2d, p=None)                                
        elif sample_mode == 'uniformsampler2':
            alpha = utils.sampler.UniformSampler2(w1.size(0), w1.size(1), is_2d, p=None)
        elif sample_mode == 'betasampler':
            alpha = utils.sampler.BetaSampler(w1.size(0), w1.size(1), is_2d, p=None, beta_alpha = self._args.beta_alpha)

        w_mixed = alpha*w1 + (1.-alpha)*w2
        y_mixed = alpha*y1 + (1.-alpha)*y2

        return w_mixed,y_mixed

    def __MaskMixup2__(self,w1,w2,sample_mode,y1,y2):                                                             

        """
        w1.shape torch.Size([bs, 512])
        y1.shape torch.Size([bs, 10])
        """

        is_2d = True if len(w1.size()) == 2 else False
        if sample_mode == 'bernoullisampler':
            # print('sample_mode = bernoullisampler, samll variance !')
            m = utils.sampler.BernoulliSampler(w1.size(0), w1.size(1), is_2d, p=None)
        elif sample_mode == 'bernoullisampler2':
            # print('sample_mode = bernoullisampler2, big variance !')
            m = utils.sampler.BernoulliSampler2(w1.size(0), w1.size(1), is_2d, p=None)

        lam = []
        for i in range(len(m)):
            lam_i = (torch.nonzero(m[i]).size(0)) / m.size(1)
            lam.append(lam_i)

        lam = np.asarray(lam)
        lam = torch.tensor(lam).unsqueeze(1)

        m1 = m.cpu()
        m2 = (1.-m).cpu()
        lam1 = lam.cpu()
        lam2 = (1.-lam).cpu() 

        w1 = w1.cpu()
        w2 = w2.cpu()        
        y1 = y1.cpu()
        y2 = y2.cpu()

        w_mixed = m1*w1 + m2*w2
        y_mixed = lam1*y1 + lam2*y2

        return w_mixed.cpu(),y_mixed.cpu()

    def __BaseMixup3__(self,w1,w2,w3,sample_mode,y1,y2,y3):

        """
        w1.shape torch.Size([4, 512])
        y1.shape torch.Size([4, 10])
        """

        is_2d = True if len(w1.size()) == 2 else False                                                                     

        if  sample_mode =='dirichletsampler':
            # raise error
            alpha = utils.sampler.DirichletSampler(w1.size(0), w1.size(1), is_2d, dirichlet_gama = self._args.dirichlet_gama)

        alpha1 = alpha[:, 0:1].cuda()
        alpha2 = alpha[:, 1:2].cuda()
        alpha3 = alpha[:, 2:3].cuda()
        
        w1 = w1.cuda()
        w2 = w2.cuda()
        w3 = w3.cuda()
        y1 = y1.cuda()
        y2 = y2.cuda()
        y3 = y3.cuda()

        w_mixed = alpha1 * w1 + alpha2 * w2 + alpha3 * w3
        y_mixed = alpha1 * y1 + alpha2 * y2 + alpha3 * y3

        return w_mixed.cpu(),y_mixed.cpu()

    def __MaskMixup3__(self,w1,w2,w3,sample_mode,y1,y2,y3):

        is_2d = True if len(w1.size()) == 2 else False
        if sample_mode == 'bernoullisampler3':
            m = utils.sampler.BernoulliSampler3(w1.size(0), w1.size(1), is_2d)

        m1 = m[:, 0:1, :].squeeze(1).cuda()
        m2 = m[:, 1:2, :].squeeze(1).cuda()
        m3 = m[:, 2:3, :].squeeze(1).cuda()
        

        """
        m1.shape: torch.Size([4, 512])
        m2.shape: torch.Size([4, 512])
        m3.shape: torch.Size([4, 512])
        """

        lam1 = []
        for i in range(len(m1)):        #bs
            lam1_i = torch.nonzero(m1[i]).size(0) / m.size(2)
            lam1.append(lam1_i)
        lam1 = np.asarray(lam1)         #   (4)
        lam1 = torch.tensor(lam1).unsqueeze(1).cuda()  #   [4,1]

        lam2 = []
        for i in range(len(m2)):        #bs
            lam2_i = torch.nonzero(m2[i]).size(0) / m.size(2)
            lam2.append(lam2_i)
        lam2 = np.asarray(lam2)         #   (4)
        lam2 = torch.tensor(lam2).unsqueeze(1).cuda()  #   [4,1]

        lam3 = []
        for i in range(len(m3)):        #bs
            lam3_i = torch.nonzero(m3[i]).size(0) / m.size(2)
            lam3.append(lam3_i)
        lam3 = np.asarray(lam3)         #   (4)
        lam3 = torch.tensor(lam3).unsqueeze(1).cuda()  #   [4,1]

        w1 = w1.cuda()
        w2 = w2.cuda()
        w3 = w3.cuda()
        y1 = y1.cuda()
        y2 = y2.cuda()
        y3 = y3.cuda()

        w_mixed = m1*w1 + m2*w2 +m3*w3
        y_mixed = lam1*y1 + lam2*y2 +lam3*y3

        return w_mixed.cpu(),y_mixed.cpu()

    def __AdversarialMixup2__(self,ws1,ws2,sample_mode):
        print('AdversarialMixup2')

    def __TwoMixup__(self,opt, exp_result_dir):

        device = torch.device('cuda')
        projected_w1_x = np.load(opt.projected_w1)['w']
        projected_w1_x = torch.tensor(projected_w1_x, device=device)      
        projected_w2_x = np.load(opt.projected_w2)['w']
        projected_w2_x = torch.tensor(projected_w2_x, device=device)     
        projected_w_set_x = torch.cat((projected_w1_x,projected_w2_x),dim=0)


        projected_w1_y = np.load(opt.projected_w1_label)['w']
        projected_w1_y = torch.tensor(projected_w1_y, device=device)      
        projected_w2_y = np.load(opt.projected_w2_label)['w']
        projected_w2_y = torch.tensor(projected_w2_y, device=device)     

        projected_w_set_y = torch.cat((projected_w1_y,projected_w2_y),dim=0)
        projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                            
        interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

    def __ThreeMixup__(self,opt, exp_result_dir):
        print("flag: ThreeMixup")

        device = torch.device('cuda')
        projected_w1_x = np.load(opt.projected_w1)['w']
        projected_w1_x = torch.tensor(projected_w1_x, device=device)      
        projected_w2_x = np.load(opt.projected_w2)['w']
        projected_w2_x = torch.tensor(projected_w2_x, device=device)   
        projected_w3_x = np.load(opt.projected_w3)['w']
        projected_w3_x = torch.tensor(projected_w3_x, device=device)            

        projected_w_set_x = torch.cat((projected_w1_x,projected_w2_x,projected_w3_x),dim=0)
        print("projected_w_set_x.shape：",projected_w_set_x.shape)               
        projected_w1_y = np.load(opt.projected_w1_label)['w']
        projected_w1_y = torch.tensor(projected_w1_y, device=device)      
        projected_w2_y = np.load(opt.projected_w2_label)['w']
        projected_w2_y = torch.tensor(projected_w2_y, device=device)     
        projected_w3_y = np.load(opt.projected_w3_label)['w']
        projected_w3_y = torch.tensor(projected_w3_y, device=device)     

        projected_w_set_y = torch.cat((projected_w1_y,projected_w2_y,projected_w3_y),dim=0)
        print("projected_w_set_y.shape：",projected_w_set_y.shape)              
        projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                            
        interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

    def __DatasetMixup__(self,opt,exp_result_dir):
        file_dir=os.listdir(opt.projected_dataset)                                                                        
        file_dir.sort()                                                                                                       

        npzfile_name = []
        for name in file_dir:                                                                                                   
            if os.path.splitext(name)[-1] == '.npz':
                npzfile_name.append(name)                                                                                   
        projected_w_npz_paths =[]
        label_npz_paths = []
        for name in npzfile_name:
            if name[-15:-4] == 'projected_w': 
                projected_w_npz_paths.append(f'{opt.projected_dataset}/{name}')

            elif name[-9:-4] == 'label':
                label_npz_paths.append(f'{opt.projected_dataset}/{name}')

        if opt.mix_w_num == 2:
            print("flag: DatasetTwoMixup")
        
        elif opt.mix_w_num == 3:
            print("flag: DatasetThreeMixup")
        
        else:
            raise Exception('please input valid w_num: 2 or 3')

        device = torch.device('cuda')

        projected_w_set_x = []       
        for projected_w_path in projected_w_npz_paths:                                                                                   
            w = np.load(projected_w_path)['w']
            w = torch.tensor(w, device=device)                                                                                 
            w = w[-1]                                                                                       #   w.shape: torch.Size([1, 8,512]))         
            projected_w_set_x.append(w)                                                                                         
        projected_w_set_x = torch.stack(projected_w_set_x)           
        projected_w_set_y = []       
        for label_npz_path in label_npz_paths:                                                                                  
            y = np.load(label_npz_path)['w']
            y = torch.tensor(y, device=device)                                                                                 
            y = y[-1]                                                                                       #   y.shape: torch.Size([1, 8]))
            projected_w_set_y.append(y)
        projected_w_set_y = torch.stack(projected_w_set_y)           
                                                                                                            #   projected_w_set_y.shape: torch.Size([38, 10])
        projected_w_set_y = torch.nn.functional.one_hot(projected_w_set_y, opt.n_classes).float().to(device)                           
                                                                                                            #   projected_w_set_y.shape: torch.Size([38, 10, 10])
        projected_w_set_x = projected_w_set_x.cpu()
        projected_w_set_y = projected_w_set_y.cpu()

        interpolated_w_set, interpolated_y_set = self.__getmixededwy__(opt, projected_w_set_x,projected_w_set_y,exp_result_dir)

        return interpolated_w_set, interpolated_y_set

    def genxyset(self):
        return self.generated_x_set, self.generated_y_set
        
    def generate(self, exp_result_dir, interpolated_w_set = None, interpolated_y_set = None):
        self._exp_result_dir = exp_result_dir
        generated_x_set, generated_y_set = self.__generatemain__(self._args, self._exp_result_dir, interpolated_w_set, interpolated_y_set)
        self.generated_x_set = generated_x_set
        self.generated_y_set = generated_y_set

    def __generatemain__(self, opt, exp_result_dir, interpolated_w_set, interpolated_y_set):
        # print("running generate main()..............")

        if interpolated_w_set is not None:
            # print("Generate mixed samples from mixed projectors numpy ndarray !")
            self.interpolated_w_set = interpolated_w_set
            self.interpolated_y_set = interpolated_y_set
            generated_x_set, generated_y_set = self.__generatefromntensor__()

        else:
            print("Generate mixed samples from mixed projectors npz files !")

            if opt.mixed_dataset != None:
                generated_x_set, generated_y_set = self.__generate_dataset__(opt, exp_result_dir)
            
            elif opt.mixed_dataset == None:
                if opt.projected_w is not None:
                    generated_x_set, generated_y_set = self.__generate_images__(
                        ctx = click.Context,                                                                                     
                        network_pkl = opt.gen_network_pkl,

                        seeds = [500, 501, 502, 503, 504, 505],
                        truncation_psi = opt.truncation_psi,
                        noise_mode = opt.noise_mode,
                        outdir = exp_result_dir,
                        class_idx = opt.class_idx,
                        projected_w = opt.projected_w,
                        mixed_label_path = None
                        # mixed_label_path = opt.projected_w_label
                    )
                elif opt.projected_w is None:
                    print("opt.generate_seeds:",opt.generate_seeds)
                    generated_x_set, generated_y_set = self.__generate_images__(
                        ctx = click.Context,                                                                             
                        network_pkl = opt.gen_network_pkl,
                        seeds = opt.generate_seeds,
                        truncation_psi = opt.truncation_psi,
                        noise_mode = opt.noise_mode,
                        outdir = exp_result_dir,
                        class_idx = opt.class_idx,
                        projected_w = opt.projected_w,
                        mixed_label_path = None
                    )              
                    raise error      
            
        return generated_x_set, generated_y_set

    def __generatefromntensor__(self):
        exp_result_dir = self._exp_result_dir

        opt = self._args

        exp_result_dir = os.path.join(exp_result_dir,f'generate-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)    

        interpolated_w_set = self.interpolated_w_set
        interpolated_y_set = self.interpolated_y_set

        if self._args.defense_mode == 'rmt':

            generated_x_set, generated_y_set = self.__getgeneratedbatchxy__(    
                    ctx = click.Context, 
                    network_pkl = opt.gen_network_pkl,
                    seeds = [500, 501, 502, 503, 504, 505],
                    truncation_psi = opt.truncation_psi,
                    noise_mode = opt.noise_mode,
                    outdir = exp_result_dir,
                    class_idx = opt.class_idx,
                    interpolated_w = interpolated_w_set,
                    interpolated_y = interpolated_y_set,
                )

        else:
                
            generated_x_set = []
            generated_y_set = []

            for i in range(len(interpolated_w_set)):
                generated_x, generated_y = self.__imagegeneratefromwset__(      
                    ctx = click.Context,
                    network_pkl = opt.gen_network_pkl,
                    seeds = [500, 501, 502, 503, 504, 505],
                    truncation_psi = opt.truncation_psi,
                    noise_mode = opt.noise_mode,
                    outdir = exp_result_dir,
                    class_idx = opt.class_idx,
                    interpolated_w = interpolated_w_set[i],
                    interpolated_y = interpolated_y_set[i],
                    interpolated_w_index = i
                )

                generated_x_set.append(generated_x)
                generated_y_set.append(generated_y)

        return generated_x_set, generated_y_set

    def __getgeneratedbatchxy__(self,                                 
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        class_idx: Optional[int],
        interpolated_w: torch.tensor,
        interpolated_y: torch.tensor     
    ):
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)                                                              

        if interpolated_w is not None:
            
            ws = interpolated_w                                            
            mixed_label = interpolated_y    
            mixed_label = mixed_label[:,0,:].squeeze(1)
            assert ws.shape[1:] == (G.num_ws, G.w_dim)                                                                    
            generated_x = []
            for _, w in enumerate(ws):
                w = w.cuda()
                img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode).cpu()
                generated_x.append(img)         
            generated_x = torch.cat(generated_x,dim=0)
            generated_y = mixed_label
            
            return generated_x, generated_y

    def __imagegeneratefromwset__(self,                                
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        class_idx: Optional[int],
        interpolated_w: torch.tensor,
        interpolated_y: torch.tensor,      
        interpolated_w_index: int                                     
    ):

        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)                                        

        if interpolated_w is not None:
            
            ws = interpolated_w.unsqueeze(0)                                             
            mixed_label = interpolated_y.unsqueeze(0)        
            mixed_label = mixed_label[-1]
            mixed_label = mixed_label[-1].unsqueeze(0)                                                         
            _, w1_label_index = torch.max(mixed_label, 1)                  

            if self._args.mix_w_num == 2:
                modified_mixed_label = copy.deepcopy(mixed_label)
                modified_mixed_label[0][w1_label_index] = 0                             

                if torch.nonzero(modified_mixed_label[0]).size(0) == 0:
                    w2_label_index = w1_label_index
                else:
                    _, w2_label_index = torch.max(modified_mixed_label, 1)

                classification = self.__labelnames__()
                assert ws.shape[1:] == (G.num_ws, G.w_dim)                                                                  
                for _, w in enumerate(ws):
                    img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
                    generated_x = img[-1]            
                    generated_y = mixed_label[-1]
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    w1_label_name = f"{classification[int(w1_label_index)]}"
                    w2_label_name = f"{classification[int(w2_label_index)]}"

                    if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
                        img_pil = PIL.Image.fromarray(img, 'RGB')

                    elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
                        img = img.transpose([2, 0, 1])
                        img = img[0]
                        img_pil = PIL.Image.fromarray(img, 'L')
                    img = img_pil.save(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed-image.png') 

                    if self._args.defense_mode != 'rmt':
                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed-image.npz', w = generated_x.cpu().numpy())                                               
                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}-mixed-label.npz', w = generated_y.cpu().numpy())                                               
            elif self._args.mix_w_num == 3:
            
                modified_mixed_label = copy.deepcopy(mixed_label)
                modified_mixed_label[0][w1_label_index] = 0                             

                if torch.nonzero(modified_mixed_label[0]).size(0) == 0:
                    w2_label_index = w1_label_index
                    w3_label_index = w1_label_index

                else:
                    _, w2_label_index = torch.max(modified_mixed_label, 1)
                    # print(f'w2_label_index = {int(w2_label_index)}')

                    modified2_mixed_label = copy.deepcopy(modified_mixed_label)
                    modified2_mixed_label[0][w2_label_index] = 0
                    
                    if torch.nonzero(modified2_mixed_label[0]).size(0) == 0:
                        w3_label_index = w2_label_index
                    else:
                        _, w3_label_index = torch.max(modified2_mixed_label, 1)
                classification = self.__labelnames__()

                assert ws.shape[1:] == (G.num_ws, G.w_dim)                                                                    
                for _, w in enumerate(ws):

                    img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)

                    generated_x = img[-1]            
                    generated_y = mixed_label[-1]
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    w1_label_name = f"{classification[int(w1_label_index)]}"
                    w2_label_name = f"{classification[int(w2_label_index)]}"
                    w3_label_name = f"{classification[int(w3_label_index)]}"

                    if self._args.dataset != 'kmnist' and self._args.dataset != 'mnist':
                        img_pil = PIL.Image.fromarray(img, 'RGB')
                    elif self._args.dataset == 'kmnist' or self._args.dataset == 'mnist':
                        img = img.transpose([2, 0, 1])
                        img = img[0]
                        img_pil = PIL.Image.fromarray(img, 'L')                                      
                    img = img_pil.save(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}+{int(w3_label_index)}-{w3_label_name}-mixed-image.png')                                        
                    if self._args.defense_mode != 'rmt':

                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}+{int(w3_label_index)}-{w3_label_name}-mixed-image.npz', w = generated_x.cpu().numpy())                                        
                        np.savez(f'{outdir}/{interpolated_w_index:08d}-{int(w1_label_index)}-{w1_label_name}+{int(w2_label_index)}-{w2_label_name}+{int(w3_label_index)}-{w3_label_name}-mixed-label.npz', w = generated_y.cpu().numpy())                                             

            return generated_x, generated_y
            

        if seeds is None:
            ctx.fail('--seeds option is required when not using --projected_w')

        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png') 

        generated_x = img[0]
        generated_y = label
        return generated_x, generated_y

    def __generate_dataset__(self, opt, exp_result_dir):
        exp_result_dir = os.path.join(exp_result_dir,f'generate-{opt.dataset}-trainset')
        os.makedirs(exp_result_dir,exist_ok=True)   

        file_dir=os.listdir(opt.mixed_dataset)
        file_dir.sort()

        mixed_projector_filenames=[]                                                                                            
        mixed_label_filenames=[]                                                                                          
        
        for name in file_dir:
            # print(name)
            if name[-15:-4] == 'projected_w':
                # print(name[-15:-4])
                mixed_projector_filenames.append(name)
            elif name[-9:-4] == 'label':
                # print(name[-9:-4])
                mixed_label_filenames.append(name)

        mixed_projector_path = []
        mixed_label_path = []
        for name in mixed_projector_filenames:
            mixed_projector_path.append(f'{opt.mixed_dataset}/{name}')

        for name in mixed_label_filenames:
            mixed_label_path.append(f'{opt.mixed_dataset}/{name}')

        generated_x_set = []
        generated_y_set = []
     
        for i in range(len(mixed_projector_path)):

            generated_x, generated_y = self.__generate_images__(
                ctx = click.Context,
                network_pkl = opt.gen_network_pkl,
                seeds = [500, 501, 502, 503, 504, 505],
                truncation_psi = opt.truncation_psi,
                noise_mode = opt.noise_mode,
                outdir = exp_result_dir,
                class_idx = opt.class_idx,
                projected_w = mixed_projector_path[i],
                mixed_label_path = mixed_label_path[i]
            )

            generated_x_set.append(generated_x)
            generated_y_set.append(generated_y)

        return generated_x_set, generated_y_set
    
    def __generate_images__(self,
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
        outdir: str,
        class_idx: Optional[int],
        projected_w: Optional[str],
        mixed_label_path:Optional[str]                                                                                          #   maggie add
    ):

        print('Loading networks from "%s"...' % network_pkl)
        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)                                                                  #   type: ignore

        if projected_w is not None:
            print(f'Generating images from projected W "{projected_w}"')        
            ws = np.load(projected_w)['w']
            ws = torch.tensor(ws, device=device)
            mixed_label = np.load(mixed_label_path)['w']
            mixed_label = torch.tensor(mixed_label, device=device)                                                              #   pylint: disable=not-callable
            mixed_label = mixed_label[-1]
         
            assert ws.shape[1:] == (G.num_ws, G.w_dim)                                                               
            for idx, w in enumerate(ws):
                img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
 
                generated_x = img[-1]                 
                generated_y = mixed_label[-1]
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                w_name= re.findall(r'/home/maggie/mmat/.*/(.*?)\+(.*?)-mixed_projected_w.npz',projected_w)                      
                w1_name =str(w_name[0][0])
                w2_name = str(w_name[0][1])

                img_pil = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
       
                img = img_pil.save(f'{outdir}/{w1_name}+{w2_name}-mixed-image.png')                                    
               
                label_path = f'{outdir}/{w1_name}+{w2_name}-mixed-label.npz'
                                
                np.savez(label_path, w = mixed_label.unsqueeze(0).cpu().numpy())                                             
              
                np.savez(f'{outdir}/{w1_name}+{w2_name}-mixed-image.npz', w = generated_x.cpu().numpy())       
                np.savez(f'{outdir}/{w1_name}+{w2_name}-mixed-label.npz', w = generated_y.cpu().numpy())         
                
            return generated_x, generated_y

        if seeds is None:
            ctx.fail('--seeds option is required when not using --projected_w')

        # Labels.
        label = torch.zeros([1, G.c_dim], device=device)
        if G.c_dim != 0:
            if class_idx is None:
                ctx.fail('Must specify class label with --class when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print ('warn: --class=lbl ignored when running on an unconditional network')

        # Generate images.
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            _, _, _, channel_num = img.shape
            
            assert channel_num in [1, 3]
            if channel_num == 1:
                PIL.Image.fromarray(img[0][:, :, 0].cpu().numpy(), 'L').save(f'{outdir}/seed{seed:04d}.png') 
            if channel_num == 3:
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png') 

        generated_x = img[0]
        generated_y = label
        return generated_x, generated_y

    def __num_range__(self, s: str) -> List[int]:
        '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
        range_re = re.compile(r'^(\d+)-(\d+)$')
        m = range_re.match(s)
        if m:
            return list(range(int(m.group(1)), int(m.group(2))+1))
        vals = s.split(',')
        return [int(x) for x in vals]
    
