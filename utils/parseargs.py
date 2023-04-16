

import argparse
from re import L
from utils.separatelist import CommaSeparatedList
import click
from typing import Optional
from typing import List    
import datetime
from utils.runid import GetRunID     
import os
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='parse command')
    subparsers = parser.add_subparsers(description='parse subcommand',dest='subcommand')
    
    parser_load = subparsers.add_parser('load',help = 'run command from a ./experiments/yaml file')
    parser_load.add_argument('--config',type=str,default='experiments/experiment-command.yaml')

    parser_run = subparsers.add_parser('run',help = 'run command from the command line')

    for parser_object in [parser_load,parser_run]:
        parser_object.add_argument('--exp_name',type=str,default=None,help='Name of the experiment',
            choices=[
                'gan-mnist','gan-kmnist','gan-cifar10','gan-cifar100','gan-lsun','gan-imagenet','gan-imagenetmixed10',
                'acgan-mnist','acgan-kmnist','acgan-cifar10','acgan-cifar100','acgan-lsun','acgan-imagenet','acgan-imagenetmixed10',
                'aae-mnist','aae-kmnist','aae-cifar10','aae-cifar100','aae-lsun','aae-imagenet','aae-imagenetmixed10',
                'vae-mnist','vae-kmnist','vae-cifar10','vae-cifar100','vae-lsun','vae-imagenet','vae-imagenetmixed10',
                'stylegan2-mnist','stylegan2-kmnist','stylegan2-cifar10','stylegan2-cifar100','stylegan2-lsun','stylegan2-imagenet','stylegan2-imagenetmixed10',
                'stylegan2ada-mnist','stylegan2ada-kmnist','stylegan2ada-cifar10','stylegan2ada-cifar100','stylegan2ada-lsun','stylegan2ada-imagenet','stylegan2ada-imagenetmixed10','stylegan2ada-svhn','stylegan2ada-stl10',
                'resnet50-mnist', 'resnet50-kmnist','resnet50-cifar10','resnet50-cifar100','resnet50-lsun','resnet50-imagenet','resnet50-imagenetmixed10','resnet50-svhn','resnet50-stl10',
                'vgg19-mnist','vgg19-kmnist', 'vgg19-cifar10','vgg19-cifar100', 'vgg19-lsun','vgg19-imagenet', 'vgg19-imagenetmixed10','vgg19-svhn','vgg19-stl10',
                'alexnet-mnist','alexnet-kmnist', 'alexnet-cifar10','alexnet-cifar100', 'alexnet-lsun','alexnet-imagenet', 'alexnet-imagenetmixed10','alexnet-svhn','alexnet-stl10',
                'densenet169-mnist','densenet169-kmnist', 'densenet169-cifar10','densenet169-cifar100', 'densenet169-lsun','densenet169-imagenet', 'densenet169-imagenetmixed10','densenet169-svhn','densenet169-stl10',
                'inception_v3-mnist','inception_v3-kmnist', 'inception_v3-cifar10','inception_v3-cifar100', 'inception_v3-lsun','inception_v3-imagenet', 'inception_v3-imagenetmixed10','inception_v3-svhn','inception_v3-stl10',
                'resnet34-mnist','resnet34-kmnist','resnet34-mnist','resnet34-cifar10','resnet34-cifar100','resnet34-lsun','resnet34-imagenet','resnet34-imagenetmixed10','resnet34-svhn','resnet34-stl10',
                'resnet18-mnist','resnet18-kmnist','resnet18-mnist','resnet18-cifar10','resnet18-cifar100','resnet18-lsun','resnet18-imagenet','resnet18-imagenetmixed10','resnet18-svhn','resnet18-stl10',
                'googlenet-mnist','googlenet-kmnist','googlenet-mnist','googlenet-cifar10','googlenet-cifar100','googlenet-lsun','googlenet-imagenet','googlenet-imagenetmixed10','googlenet-svhn','googlenet-stl10',
                'preactresnet18-kmnist', 'preactresnet18-svhn', 'preactresnet18-cifar10','preactresnet18-imagenetmixed10',
                'preactresnet34-kmnist', 'preactresnet34-svhn', 'preactresnet34-cifar10','preactresnet34-imagenetmixed10',
                'preactresnet50-kmnist', 'preactresnet50-svhn', 'preactresnet50-cifar10','preactresnet50-imagenetmixed10',
                'wideresnet2810-kmnist', 'wideresnet2810-svhn', 'wideresnet2810-cifar10'

            ]  

        )
        parser_object.add_argument('--cla_model',type=str,default=None,
            choices=['resnet34','resnet50', 'vgg19','alexnet','densenet169','inception_v3','resnet18','googlenet','preactresnet18','preactresnet34','preactresnet50','wideresnet2810']
        )

        parser_object.add_argument('--gen_model',type=str,default=None,
            choices=['gan', 'acgan', 'aae', 'vae','stylegan2','stylegan2ada']
        )

        parser_object.add_argument('--dataset',type=str,default=None,
            choices=['mnist', 'kmnist', 'cifar10', 'cifar100', 'lsun', 'imagenet','imagenetmixed10','svhn','stl10'],
            help="""
            mnist:      size = 1x28x28, class_num = 10, train_sample_num = 60000 (6000 per class), test_sample_num = 10000 (1000 per class), link = http://yann.lecun.com/exdb/mnist/
            kmnist:     size = 1x28x28, class_num = 10, train_sample_num = 60000 (6000 per class), test_sample_num = 10000 (1000 per class), link = https://github.com/rois-codh/kmnist
            cifar10:    size = 3x32x32, class_num = 10, train_sample_num = 50000 (5000 per class), test_sample_num = 10000 (1000 per class), link :https://www.cs.toronto.edu/~kriz/cifar.html
            cifar100:   size = 3x32x32, class_num = 100, train_sample_num = 50000 (500 per class), test_sample_num = 10000 (100 per class), link :https://www.cs.toronto.edu/~kriz/cifar.html
            lsun:       size = 3x256x256, class_num = 30, train_sample_num = ???, test_sample_num = ???, link: https://www.yf.io/p/lsun
            imagenet:   size = 1x1024x1024, class_num = 1000, train_sample_num = 1321167, test_sample_num = 100000, link: http://image-net.org/
            imagenetmixed10: size=1x256x256, class_num = 10, train_sample_num = 77237, test sample_num = 3000
            stl10 :     size = 3x96x96(改成了3x128x128),     class_num = 10
            svhn:      size = 3x32x32, class_num = 10, train_sample_num = 73257  (6000 per class), test_sample_num = 26032 (1000 per class), link = http://ufldl.stanford.edu/housenumbers/ 
            
            """
        )
        parser_object.add_argument('--mode',type=str,default='train',
            choices=['train','test','attack','defense','valid','project','generate','interpolate','defense'],
            help="""
                train:          standard training   
                test:           evaluate accuracy on clean testing dataset
                attack:         generate adversarial examples and attack classifiers
                valid:          validing
                project:        project x to w using trained network  
                generate:       generate x from project vector w
                interpolate:    generate interpolation of w
                """            
        )
        parser_object.add_argument('--n_classes',type=int,default=10,help='Number of classes of the dataset')
        parser_object.add_argument('--batch_size',type=int,default=32,help='Batch size of the dataset setting for the training process')
        parser_object.add_argument('--epochs',type=int,default=20,help='Epochs number setting for the training process')
        parser_object.add_argument('--lr',type=float,default=0.01,help='Learning rate setting for the training process')
        parser_object.add_argument('--save_path',type=str,default='/home/data/maggie/result-newhome',help='Output path for saving results')
        parser_object.add_argument('--cpus',type=int,default=4,help='Number of CPUs to use')
        parser_object.add_argument('--gpus',type=int,default=1,help='Number of GPUS to use')

        #-------------------------arguments for stylegan2ada train-------------------------
        parser_object.add_argument('--metrics', type=CommaSeparatedList(), default='fid50k_full',help='Comma-separated list or "none"', )
        parser_object.add_argument('--conditional',type=bool,default=False,help='Train conditional model based on dataset labels')
        parser_object.add_argument('--dry-run', help='Print training options and exit', type=bool, default=False)
        parser_object.add_argument('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT', default=20)
        
        # Dataset.
        parser_object.add_argument('--data', help='Training data (directory or zip)', metavar='PATH')
        parser_object.add_argument('--subset', help='Train with only N images [default: all]', type=int, metavar='INT') #defualt=all
        parser_object.add_argument('--mirror',type=bool,default=False,help='Enable dataset x-flips')
        
        # Base config.
        parser_object.add_argument('--cfg', type=click.Choice(['auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar']), default='auto',help='Base config')
        parser_object.add_argument('--gamma', type=float,help='Override R1 gamma', )
        parser_object.add_argument('--kimg', type=int,help='Override training duration')
        
        # Discriminator augmentation.
        parser_object.add_argument('--aug', help='Augmentation mode [default: ada]', type=click.Choice(['noaug', 'ada', 'fixed']), default='ada')
        parser_object.add_argument('--p', help='Augmentation probability for --aug=fixed', type=float)
        parser_object.add_argument('--target', help='ADA target value for --aug=ada', type=float)
        parser_object.add_argument('--augpipe', help='Augmentation pipeline [default: bgc]', type=click.Choice(['blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc', 'bgcf', 'bgcfn', 'bgcfnc']), default='bgc')
        
        # Transfer learning.
        parser_object.add_argument('--resume', help='Resume training [default: noresume]', metavar='PKL', default='noresume')
        parser_object.add_argument('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')
        
        # Performance options.
        parser_object.add_argument('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
        parser_object.add_argument('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
        parser_object.add_argument('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
        parser_object.add_argument('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
        parser_object.add_argument('--pretrain_pkl_path', help = 'pretrained stylegan2ada model path',type = str, default=None)

        #-------------------------arguments for stylegan2ada projector-------------------------
        parser_object.add_argument('--gen_network_pkl', help='Network pickle filename',default = None)
        parser_object.add_argument('--target_fname', help='Target image file to project to', metavar='FILE', default= None)
        parser_object.add_argument('--num_steps', help='Number of optimization steps', type=int, default=1000)         
        parser_object.add_argument('--save_video', help='Save an mp4 video of optimization progress', type=bool, default=False)
        parser_object.add_argument('--target_dataset', help = 'The zip dataset path of target png images to project to', metavar='PATH',type = str, default = None)
        parser_object.add_argument('--viewdataset_path', help = 'The png dataset path of target png images to project to', metavar='PATH',type = str, default = None)
        parser_object.add_argument("--project_target_num",type = int, help = 'The number of target image to project to', default= None  )
        
        #-------------------------arguments for stylegan2ada generate-------------------------
        parser_object.add_argument('--truncation_psi', type=float, help='Truncation psi', default=1)
        parser_object.add_argument('--class_idx', type=int, help='Class label (unconditional if not specified)')
        parser_object.add_argument('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const')
        parser_object.add_argument('--projected_w', help='Projection result file', type=str, metavar='FILE',default = None)
        parser_object.add_argument('--mixed_dataset', help='Projection result file', type=str, metavar='FILE',default = None)            
        # parser_object.add_argument('--generate_seeds', type=Optional[List[int]], help='List of random generate seeds')
        parser_object.add_argument('--generate_seeds', nargs='+', type=int, help='List of random generate seeds',default = None, )
        parser_object.add_argument('--projected_w_label', help='Projection result file', type=str, metavar='FILE',default = None)
        
        #-------------------------arguments for stylegan2ada interpolation-------------------------
        parser_object.add_argument('--projected_w1', help='Projection result file', type=str, metavar='FILE')
        parser_object.add_argument('--projected_w2', help='Projection result file', type=str, metavar='FILE')
        parser_object.add_argument('--projected_w3', help='Projection result file', type=str, metavar='FILE',default= None)
        parser_object.add_argument('--projected_w1_label', help='Projection result file', type=str, metavar='FILE')
        parser_object.add_argument('--projected_w2_label', help='Projection result file', type=str, metavar='FILE')
        parser_object.add_argument('--projected_w3_label', help='Projection result file', type=str, metavar='FILE',default= None)

        parser_object.add_argument('--mix_mode', help='mix mode of the projected w', type=str, default='basemixup', choices=['basemixup', 'maskmixup', 'adversarialmixup'])
        parser_object.add_argument('--mix_w_num', help='number of the projected w for mixup', type=int, default=2)
        parser_object.add_argument('--sample_mode', help='share alpha for projected_w.size(1) or not', type=str, default='betasampler',choices=['uniformsampler', 'uniformsampler2', 'bernoullisampler','bernoullisampler2', 'betasampler', 'dirichletsampler','bernoullisampler3'])
        parser_object.add_argument('--projected_dataset', help = 'The projected w dataset path of target png images to interpolate', type = str, default = None)
        parser_object.add_argument('--mix_img_num', help='number of the mixed images', type=int, default=None)
        parser_object.add_argument('--beta_alpha', help='beta(alpha,alpha)', type=float, default=1)
        parser_object.add_argument('--dirichlet_gama', help='dirichlet(gama, gama)', type=float, default=1)


        #-------------------------arguments for classifier train-------------------------
        parser_object.add_argument('--train_mode', help='standard train or adversarial train', type=str, default=None,choices=['gen-train','cla-train'])
        parser_object.add_argument('--test_mode', help='test', type=str, default='classifier-test',choices=['gen-test','classifier-test','classifier-advtest'])
        parser_object.add_argument('--pretrained_on_imagenet', help='pretrain', type= bool, default= False)
        
        #-------------------------arguments for classifier attack-------------------------
        parser_object.add_argument('--attack_mode', help='attack method', type=str, default='fgsm', 
        choices=['fgsm','deepfool','bim','cw','pgd','om-fgsm','om-pgd','fog','snow','elastic','gabor','jpeg','autoattack'])


        parser_object.add_argument('--cla_network_pkl', help='cla_network_pkl', type=str)
        parser_object.add_argument('--attack_eps', help='number of the FGSM epsilon', type=float, default=0.2)
        parser_object.add_argument('--whitebox',action='store_true', help='white box attack')
        parser_object.add_argument('--blackbox',action='store_true', help='black box attack')
        parser_object.add_argument('--latentattack', action='store_true', help='latent adversarial attack')
        parser_object.add_argument('--perceptualattack', action='store_true', help='Perceptual attack')
        parser_object.add_argument('--confidence', help='Confidence of CW adversarial examples', type=float, default=0.0)


        #-------------------------arguments for classifier defense-------------------------
        parser_object.add_argument('--defense_mode', help='defense method', type=str, default='rmt',choices=['at','mmat','rmt','inputmixup','manifoldmixup','patchmixup','puzzlemixup','cutmixup','dmat'])
        # parser_object.add_argument('--adv_dataset', help='adv_dataset', type=str)
        parser_object.add_argument('--mix_dataset', help='mix_dataset', type=str)
        parser_object.add_argument('--aug_adv_num',type=int, default=None)
        parser_object.add_argument('--aug_num',type=int, default=None)
        parser_object.add_argument('--aug_mix_rate',type=float, default=None)
        parser_object.add_argument('--train_adv_dataset', help='train_adv_dataset', type=str)            
        parser_object.add_argument('--test_adv_dataset', help='test_adv_dataset', type=str)         
        parser_object.add_argument('--train_adv_dataset_2', help='train_adv_dataset_2', type=str)      

        #-------------------------other arguments-------------------------
        parser_object.add_argument('--img_size',type=int, default=32)
        parser_object.add_argument('--channels', type=int, default=1)
        parser_object.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")    
        parser_object.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser_object.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser_object.add_argument("--sample_interval", type=int, default=2000, help="interval betwen image samples")
        parser_object.add_argument('--workers', type=int, default=4)
        parser_object.add_argument('--seed', type=int, default=0)  
        parser_object.add_argument('--save_images_every', type=int, default=10)
        parser_object.add_argument('--save_every', type=int, default=1000)
    
    return parser.parse_args()

def reset_arguments(args):
    # set dataset parameters
    if args.dataset == 'mnist':
        args.n_classes = 10;    args.img_size = 32;     args.channels = 1  
    elif args.dataset == 'kmnist':
        args.n_classes = 10;    args.img_size = 32;     args.channels = 1        
    elif args.dataset == 'cifar10':
        args.n_classes = 10;    args.img_size = 32;     args.channels = 3  
    elif args.dataset == 'cifar100':
        args.n_classes = 100;   args.img_size = 32;     args.channels = 3  
    elif args.dataset == 'imagenet':
        args.n_classes = 1000;  args.img_size = 256;   args.channels = 3  
    elif args.dataset == 'imagenetmixed10':
        args.n_classes = 10;    args.img_size = 256;   args.channels = 3  
    elif args.dataset == 'lsun':
        args.n_classes = 10;    args.img_size = 256;    args.channels = 3  
    elif args.dataset == 'svhn':
        args.n_classes = 10;    args.img_size = 32;     args.channels = 3  
    elif args.dataset == 'stl10':
        args.n_classes = 10;    args.img_size = 64;     args.channels = 3            

    return args

def check_arguments(args):
    if args.mode == None:                                                                        
        raise Exception('args.mode=%s,invalid mode,please input the mode' % args.mode)
    if args.save_path == None:
        raise Exception('args.save_path=%s,please input the save_path' % args.save_path)
    if  args.exp_name == None:
        raise Exception('args.exp_name=%s,please input the exp_name' % args.exp_name)    

def set_exp_result_dir(args):

    if args.seed == 0:
        save_path = args.save_path                                                                            
    else:
        save_path = f'{args.save_path}/{args.seed}'

    cur=datetime.datetime.now()
    date = f'{cur.year:04d}{cur.month:02d}{cur.day:02d}'
    print("date:",date)

    if args.whitebox == True:
        attack = "whitebox"
        print("whitebox attack")

    elif args.blackbox == True:
        attack = "blackbox"
        print("blackbox attack")
    
    if args.mode == 'train':
        exp_result_dir = f'{save_path}/{args.mode}/{args.train_mode}/{args.exp_name}/{date}'
        print("exp_result_dir:",exp_result_dir)
    elif args.mode == 'attack':
        exp_result_dir = f'{save_path}/{args.mode}/{args.attack_mode}/{args.exp_name}/{date}'   
    elif args.mode == 'project':
        exp_result_dir = f'{save_path}/{args.mode}/{args.exp_name}/{date}'             
    elif args.mode == 'interpolate':
        exp_result_dir = f'{save_path}/{args.mode}/{args.mix_w_num}mixup/{args.mix_mode}/{args.sample_mode}/{args.exp_name}/{date}'
    elif args.mode == 'defense':     
        if args.defense_mode in ["at","dmat"]:
            exp_result_dir = f'{save_path}/{args.mode}/{args.defense_mode}/{args.attack_mode}/{args.exp_name}/{attack}/{date}'
        elif args.defense_mode == "mmat":
            exp_result_dir = f'{save_path}/{args.mode}/{args.defense_mode}/{args.attack_mode}/{args.mix_mode}-{args.sample_mode}/{args.exp_name}/{date}'
        elif args.defense_mode in ['rmt','inputmixup','manifoldmixup','patchmixup','puzzlemixup','cutmixup']:
            exp_result_dir = f'{save_path}/{args.mode}/{args.defense_mode}/{args.attack_mode}/{args.mix_mode}-{args.sample_mode}/{args.exp_name}/{attack}/{date}'

    else:
        exp_result_dir=f'{save_path}/{args.mode}/{args.exp_name}/{date}'    

    cur_run_id = GetRunID(exp_result_dir)
    exp_result_dir = os.path.join(exp_result_dir, f'{cur_run_id:05d}')    

    return exp_result_dir

def correct_args_dictionary(args,args_dictionary,DO_NOT_EXPORT):
    
    if args.subcommand == 'load':                                                                               
        print('args.subcommand=%s,load from the yaml config' % args.subcommand)
        config_dictionary = yaml.load(open(args.config))   

        # normalize string format in the yaml file
        for key in config_dictionary:                                                                           
            if type(config_dictionary[key]) == str:
                if config_dictionary[key] == 'true':
                    config_dictionary[key] = True                                                         
                if config_dictionary[key] == 'false':
                    config_dictionary[key] = False                                                              
                if config_dictionary[key] == 'null':
                    config_dictionary[key] = None

        # remove keys belong to the DO_NOT_EXPORT list from the configure 
        for key in config_dictionary:
            if key not in DO_NOT_EXPORT:                                                                            
                args_dictionary[key] = config_dictionary[key]
            else:                                                                                                  
                print(f"Please ignore the keys '{key}' from the yaml file !")
        print('args_dictionary from load yaml file =%s' % args_dictionary)      

    elif args.subcommand == 'run':
        print('args.subcommand=%s, run the command line' % args.subcommand)
    elif args.subcommand == None:
        raise Exception('args.subcommand=%s,please input the subcommand !' % args.subcommand)   
    else:
        raise Exception('args.subcommand=%s,invalid subcommand,please input again !' % args.subcommand)

    return args_dictionary

def get_stylegan2ada_args(args_dictionary_copy):
    setup_training_loop_kwargs_list = [
        'gpus','snap','metrics','seed','data','cond','subset','mirror','cfg','gamma',
        'kimg','batch_size','aug','p','target','augpipe','resume','freezed','fp32','nhwc',
        'allow_tf32','nobench','workers','pretrain_pkl_path']

    stylegan2ada_config_kwargs = dict()
    for key in setup_training_loop_kwargs_list:
        if key in args_dictionary_copy:
            stylegan2ada_config_kwargs[key] = args_dictionary_copy[key]

    return stylegan2ada_config_kwargs

def copy_args_dictionary(args_dictionary,DO_NOT_EXPORT):

    args_dictionary_copy = dict(args_dictionary)                                                                   
    for key in DO_NOT_EXPORT:
        if key in args_dictionary:
            del args_dictionary_copy[key]
    return args_dictionary_copy

def main():
    args = parse_arguments()
    args_dictionary = vars(args)                                                                                  

    DO_NOT_EXPORT = ['maggie','xieldy'] 
    args_dictionary = correct_args_dictionary(args,args_dictionary,DO_NOT_EXPORT)

    args_dictionary_copy = copy_args_dictionary(args_dictionary,DO_NOT_EXPORT)
    args_dictionary_copy_yaml = yaml.dump(args_dictionary_copy)                                                 
    
    stylegan2ada_config_kwargs = get_stylegan2ada_args(args_dictionary_copy)

    check_arguments(args)
    args = reset_arguments(args)

    exp_result_dir = set_exp_result_dir(args)
    os.makedirs(exp_result_dir, exist_ok=True)
    print('Experiment result save dir: %s' % exp_result_dir)

    exp_yaml=open(f'{exp_result_dir}/experiment-command.yaml', "w")    
    exp_yaml.write(args_dictionary_copy_yaml)

    return args, exp_result_dir, stylegan2ada_config_kwargs
