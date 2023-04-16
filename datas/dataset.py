

import os
from torchvision.transforms import transforms
import torchvision.datasets
import numpy as np
import copy
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
import robustness.datasets


class MaggieMNIST(torchvision.datasets.MNIST):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace mnist data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 28, 28)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 1))
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        self.data = data

    
    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment mnist data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
      
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 28, 28)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 1)) 
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieKMNIST(torchvision.datasets.KMNIST):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace kmnist data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 28, 28)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 1))  
       
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        self.data = data


    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment kmnist data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 28, 28)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 1))  
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets
 
        self.data = data


class MaggieCIFAR10(torchvision.datasets.CIFAR10):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace cifar10 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        print('constraucting adv cifar10 testset')
        data = self.data

        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  

        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment cifar10 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data        


class MaggieCIFAR100(torchvision.datasets.CIFAR100):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace cifar100 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  
        
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        self.data = data


    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment cifar100 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
      
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieImageNet(torchvision.datasets.ImageNet):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace imagenet data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):

        data = self.data    
        
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 256, 256)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment imagenet data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 256, 256)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  
      
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieLSUN(torchvision.datasets.LSUN):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace lsun data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 256, 256)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        
        self.data = data

    def augmentdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment lsun data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 256, 256)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  
     
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data


class MaggieSVHN(torchvision.datasets.SVHN):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace svhn data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 32, 32)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        

        self.data = data

    def augdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment svhn data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 32, 32)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data    


class MaggieSTL10(torchvision.datasets.STL10):
    r'introduce this class'

    def replacedataset(self,rep_xndarray,rep_yndarray):
        self._rep_xndarray = rep_xndarray
        self._rep_yndarray = rep_yndarray
        self.__getrepdataset__()
        print("finished replace stl10 data and targets with rep_xndarray,rep_yndarray ")

    def __getrepdataset__(self):
        data = self.data
        
      
        self._rep_xndarray = np.vstack(self._rep_xndarray).reshape(-1, 3, 96, 96)
        self._rep_xndarray = self._rep_xndarray.transpose((0, 2, 3, 1))  
      
        data = data.tolist()
        data = self._rep_xndarray
        data = np.array(data)
        

        self.data = data

    def augdataset(self,aug_xndarray,aug_yndarray):
        self._aug_xndarray = aug_xndarray
        self._aug_yndarray = aug_yndarray
        self.__getaugdataset__()
        print("finished augment stl10 data and targets with aug_xndarray,aug_yndarray ")
    
    def __getaugdataset__(self):
        data = self.data
        targets = self.targets  
        self._aug_xndarray = np.vstack(self._aug_xndarray).reshape(-1, 3, 96, 96)
        self._aug_xndarray = self._aug_xndarray.transpose((0, 2, 3, 1))  
       
        data = data.tolist()
        data.extend(self._aug_xndarray)
        data = np.array(data)
        targets.extend(self._aug_yndarray)
        self.targets  = targets

        self.data = data

class RMDataset:
    def __init__(self,args, custom_traindataset = None, custom_testdataset =None) -> None:
        print(f'initilize the dataset loading parameters')
        self._args = args 
        if custom_traindataset == None:
            self._traindataset = self.__loadtraindataset__()  
            self._testdataset = self.__loadtestdataset__() 
        else:
            self._traindataset = custom_traindataset
            self._testdataset = custom_testdataset

    def traindataset(self):
        return self._traindataset
    
    def testdataset(self):
        return self._testdataset

    def __loadtraindataset__(self):
        if self._args.dataset == 'mnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size            

            os.makedirs("/home/data/maggie/mnist", exist_ok=True)
            train_dataset = MaggieMNIST(                                             
                "/home/data/maggie/mnist",
                train=True,                                            
                download=True,                                       
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset

        elif self._args.dataset == 'kmnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size            
            
            os.makedirs("/home/data/maggie/kmnist", exist_ok=True)
            train_dataset = MaggieKMNIST(                                            
                "/home/data/maggie/kmnist",
                train=True,                                            
                download=True,                                         
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            ) 
            return train_dataset

        elif self._args.dataset == 'cifar10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            print(f'load cifar10 dataset')
            os.makedirs("/home/data/maggie/cifar10", exist_ok=True)
            train_dataset = MaggieCIFAR10(                                             
                "/home/data/maggie/cifar10",
                train=True,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]
                ),
            )
            return train_dataset

        elif self._args.dataset == 'cifar100':
            os.makedirs("/home/data/maggie/cifar100", exist_ok=True)  
            
            train_dataset = MaggieCIFAR100(                                            
                "/home/data/maggie/cifar100",
                train=True,                                             
                download=True,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(self._args.img_size), 
                        transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        
                    ]
                ),
            )
            return train_dataset

        elif self._args.dataset == 'imagenet':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size 

            os.makedirs("/home/data/ImageNet", exist_ok=True)
            train_dataset = MaggieImageNet(                                             
                "/home/data/ImageNet",
                split='train',
                download=False,
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(crop_size),                       
                        transforms.CenterCrop(crop_size),                    
                        transforms.ToTensor(),                                  
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]     
                ),
            ) 
            return train_dataset

        elif self._args.dataset == 'imagenetmixed10':
            in_path = "/root/autodl-tmp/maggie/data/ImageNet"             
            in_info_path = "/root/autodl-tmp/maggie/data/ImageNet/info"
           
            in_hier = ImageNetHierarchy(in_path, in_info_path)                  

            superclass_wnid = common_superclass_wnid('mixed_10')           
            class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

            custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
            print("custom_dataset.__dict__.keys()",custom_dataset.__dict__.keys())

            train_dataset = custom_dataset
            return train_dataset

        elif self._args.dataset == 'lsun':
            os.makedirs("/home/data/maggie/lsun/20210413", exist_ok=True)
            train_dataset =MaggieLSUN(                                             
                "/home/data/maggie/lsun/20210413",
                classes=['church_outdoor_train','classroom_train','tower_train'],
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(self._args.img_size),                       
                        transforms.CenterCrop(self._args.img_size),                    
                        transforms.ToTensor(),                                  
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]     
                ),
                target_transform = None
            ) 
            return train_dataset
        
        elif self._args.dataset == 'svhn':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/maggie/svhn", exist_ok=True)
            train_dataset = MaggieSVHN(                                             
                "/home/data/maggie/svhn",
                split='train',                                             
                download=True,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                                               
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset            

        elif self._args.dataset == 'stl10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size    

            os.makedirs("/home/data/maggie/stl10", exist_ok=True)
            train_dataset = MaggieSTL10(                                             
                "/home/data/maggie/stl10",
                split='train',                                             
                download=True,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return train_dataset

    def __loadtestdataset__(self):
        if self._args.dataset == 'mnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size    

            os.makedirs("/home/data/maggie/mnist", exist_ok=True)
            test_dataset = MaggieMNIST(                                             
                "/home/data/maggie/mnist",
                train=False,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'kmnist':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size               
            os.makedirs("/home/data/maggie/kmnist", exist_ok=True)
            test_dataset = MaggieKMNIST(                                          
                "/home/data/maggie/kmnist",
                train=False,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'cifar10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/maggie/cifar10", exist_ok=True)
            test_dataset = MaggieCIFAR10(                                             
                "/home/data/maggie/cifar10",
                train=False,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                         
                        
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'cifar100':
            os.makedirs("/home/data/maggie/cifar100", exist_ok=True)
            test_dataset = MaggieCIFAR100(                                             
                "/home/data/maggie/cifar100",
                train=False,                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(self._args.img_size), 
                        transforms.CenterCrop(self._args.img_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                        
                    ]
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'imagenet':

            os.makedirs("/home/data/ImageNet", exist_ok=True)

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size 

            test_dataset = MaggieImageNet(                                             
                "/home/data/ImageNet",
                split='val',
                download=False,
                transform=transforms.Compose(                              
                    [
                        transforms.Resize(crop_size),                      
                        transforms.CenterCrop(crop_size),                
                        transforms.ToTensor(),                                               
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]     
                ),
            )
            return test_dataset    

        elif self._args.dataset == 'imagenetmixed10':
            in_path = "/root/autodl-tmp/maggie/data/ImageNet"           
            in_info_path = "/root/autodl-tmp/maggie/data/ImageNet/info"


            in_hier = ImageNetHierarchy(in_path, in_info_path)                  

            superclass_wnid = common_superclass_wnid('mixed_10')        
            class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

            custom_dataset = robustness.datasets.CustomImageNet(in_path, class_ranges)
            test_dataset = custom_dataset
            return test_dataset

        elif self._args.dataset == 'lsun':
            os.makedirs("/home/data/maggie/lsun/20210413", exist_ok=True)
            test_dataset = MaggieLSUN(                                             
                "/home/data/maggie/lsun/20210413",
                classes=['church_outdoor_test','classroom_test','tower_test'],
                transform=transforms.Compose(                               
                    [
                        transforms.Resize(self._args.img_size),                       
                        transforms.CenterCrop(self._args.img_size),                    
                        transforms.ToTensor(),                                  
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                    ]     
                ),
                target_transform = None
            )
            return test_dataset    

        elif self._args.dataset == 'stl10':

            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size                
            os.makedirs("/home/data/maggie/stl10", exist_ok=True)
            test_dataset = MaggieSTL10(                                             
                "/home/data/maggie/stl10",
                split='test',                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )  
            return test_dataset    
        
        elif self._args.dataset == 'svhn':
            if self._args.cla_model == 'inception_v3':
                crop_size = 299
            else:
                crop_size = self._args.img_size

            os.makedirs("/home/data/maggie/svhn", exist_ok=True)
            test_dataset = MaggieSVHN(                                             
                "/home/data/maggie/svhn",
                split='test',                                             
                download=False,                                          
                transform=transforms.Compose(
                    [
                        transforms.Resize(crop_size), 
                        transforms.CenterCrop(crop_size),                        
                         
                        
                        transforms.ToTensor(), 
                        transforms.Normalize([0.5], [0.5])
                    ]
                ),
            )  
            return test_dataset  

class Array2Dataset:
    def __init__(self, args, x_ndarray, y_ndarray, ori_dataset: MaggieCIFAR10):
        print(f'将{args.dataset}的对抗样本数组x_ndarray,y_ndarray变换为Dataset')
        self._args = args
        self._x_ndarray = x_ndarray
        self._y_ndarray = y_ndarray
        self._ori_dataset_4_rep = copy.deepcopy(ori_dataset)
        self._ori_dataset_4_aug = copy.deepcopy(ori_dataset)

    def repdataset(self)->"RMDataset(torchvision.datasets.__dict__)": 
        self._rep_dataset = self.__getrepdataset__()
        return self._rep_dataset

    def augdataset(self) ->"RMDataset(torchvision.datasets.__dict__)":
        self._aug_dataset = self.__getaugdataset__()
        return self._aug_dataset
    
    def __getrepdataset__(self):

        self._ori_dataset_4_rep.replacedataset(self._x_ndarray,self._y_ndarray)
        return self._ori_dataset_4_rep
    
    def __getaugdataset__(self)->"RMDataset(torchvision.datasets.__dict__)":
        self._ori_dataset_4_aug.augmentdataset(self._x_ndarray,self._y_ndarray)
        return self._ori_dataset_4_aug

