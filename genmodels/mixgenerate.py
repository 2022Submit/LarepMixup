"""
Author: maggie
Date:   2021-06-22
Place:  Xidian University
@copyright
"""

from logging import error, exception
from random import sample
from PIL.Image import RASTERIZE
from numpy.core.fromnumeric import shape
import torch
from torch.nn.functional import interpolate
import genmodels.gan
import genmodels.acgan
import genmodels.aae
import genmodels.vae
import genmodels.stylegan2
import genmodels.stylegan2ada
import numpy as np
import os

class CustomGenNet(torch.nn.Module):                                                                                         
    def __init__(self):
        super(CustomGenNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = torch.nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = torch.nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv_1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv_2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = torch.nn.functional.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

class GenModel:
    def __init__(self,args):
        self._args = args

        if self._args.gen_model == "aae":
            encoder = genmodels.aae.Encoder(self._args)
            decoder = genmodels.aae.Decoder(self._args)
            discriminator = genmodels.aae.Discriminator(self._args)               

        elif self._args.gen_model == "acgan":
            generator = genmodels.acgan.Generator(self._args)
            discriminator = genmodels.acgan.Discriminator(self._args)

        elif self._args.gen_model == "gan":
            generator = genmodels.gan.Generator(self._args)     
            discriminator = genmodels.gan.Discriminator(args) 

    def aaemodel(self):
        return 0
    
    def acganmodel(self):
        return 1

    def ganmodel(self):
        return 2 

class MixGenerate:
    r"""
        introduce this class
    """
    def __init__(self, args, exp_result_dir, stylegan2ada_config_kwargs) -> None:

        self._args = args
        self._exp_result_dir = exp_result_dir
        self._stylegan2ada_config_kwargs = stylegan2ada_config_kwargs

        if self._args.gen_model == "stylegan2ada":
            self._model = genmodels.stylegan2ada.MaggieStylegan2ada(self._args)                                                
        elif self._args.gen_model == "stylegan2":
            self._model = genmodels.stylegan2.MaggieStylegan2(self._args)                                                      

        if self._args.gen_network_pkl == None:           
            self._args.gen_network_pkl = self.__getpkl__()                                                                 

    def __getpkl__(self):
        genmodel_dict = ['gan', 'acgan', 'aae', 'vae','stylegan2','stylegan2ada']        
        if self._args.gen_model in genmodel_dict:
            gen_network_pkl = self.__getgenpkl__()
        else:
            gen_network_pkl = self.__getlocalpkl__()
        return gen_network_pkl

    def __getgenpkl__(self):
        if self._args.gen_model == "stylegan2ada":
            self._model.train(self._exp_result_dir, self._stylegan2ada_config_kwargs)
            snapshot_network_pkls = self._model.snapshot_network_pkls()                                                        
            snapshot_network_pkl = snapshot_network_pkls[-1]
        else:
            print("without gen model")
        return snapshot_network_pkl

    def __getlocalpkl__(self)->"CustomGenNet":
        local_model_pkl = "abc test"
        return local_model_pkl

    def projectmain(self, cle_train_dataloader):
        print("self._args.dataset:",self._args.dataset)
        if self._args.dataset =='cifar10' or self._args.dataset =='cifar100' or self._args.dataset =='kmnist':
            self.cle_x_train = cle_train_dataloader.dataset.data        
            self.cle_y_train = cle_train_dataloader.dataset.targets 
        elif self._args.dataset =='svhn' or self._args.dataset =='stl10':
            self.cle_x_train = cle_train_dataloader.dataset.data
            self.cle_y_train = cle_train_dataloader.dataset.labels   

        if  self._args.dataset =='imagenetmixed10':
            self.cle_y_train = cle_train_dataloader.dataset.targets
               
        """
        self.cle_y_train.type: <class 'list'>
        self.cle_y_train.len: 77237
        """                    
        sample_num = len(self.cle_y_train)
        batch_num = len(cle_train_dataloader)
        batch_size = self._args.batch_size
        print("sample_num:",sample_num)                                                                
        print("batch_num:",batch_num)                                                                   
        print("batch_size:",batch_size)                                                                 
        """
        Mixed10 training set
        sample_num: 77237
        batch_num: 2414
        batch_size: 32
        """

        if self._args.mode == "project":
            if self._args.projected_dataset == None:

                if self._args.dataset != 'imagenetmixed10':                                           
                    cle_w_train = []
                    cle_y_train = []                
                    for batch_index in range(batch_num):                                             
                        cle_x_trainbatch = self.cle_x_train[batch_index * batch_size : (batch_index + 1) * batch_size]
                        cle_y_trainbatch = self.cle_y_train[batch_index * batch_size : (batch_index + 1) * batch_size]                                                

                        print(f"Projecting *{self._args.dataset}* {batch_index}/{batch_num} batch data sets...")                   
                        pro_w_trainbatch, pro_y_trainbatch = self.__batchproject__(batch_index,cle_x_trainbatch, cle_y_trainbatch)                                        
                        cle_w_train.append(pro_w_trainbatch)                  
                        cle_y_train.append(pro_y_trainbatch)

               
                elif self._args.dataset == 'imagenetmixed10':                                         
                    cle_w_train = []
                    cle_y_train = [] 
                    print("cle_w_train.type:",type(cle_w_train))                                        
                    print("cle_y_train.type:",type(cle_y_train))    
                    """
                    cle_w_train.type: <class 'list'>
                    cle_y_train.type: <class 'list'>
                    """                                    
                    for batch_index in range(batch_num):
                        print("batch_index:",batch_index)                     
                        for batch_idx, (imgs, labs) in enumerate(cle_train_dataloader):   集                          
                            print("batch_idx:",batch_idx)

                            if batch_idx == batch_index:
                        
                                """
                                imgs.dtype: torch.float32
                                imgs.shape torch.Size([32, 3, 256, 256])
                                """              
                                imgs = imgs.numpy()
                                imgs = (imgs*255).astype(np.uint8)                  
          

                                cle_x_trainbatch = imgs
                                cle_y_trainbatch = labs.numpy().tolist()                         
                                
                                """
                                cle_x_trainbatch.type: <class 'numpy.ndarray'>
                                cle_x_trainbatch.shape: (32, 256, 256, 3)
                                cle_y_trainbatch.type: <class 'list'>
                                cle_y_trainbatch.len: 32
                                """

                                print(f"Projecting *{self._args.dataset}* {batch_index+1}/{batch_num} batch data sets...")                      
                                """Projecting *imagenetmixed10* 0/2414 batch data sets..."""
                                
                                pro_w_trainbatch, pro_y_trainbatch = self.__batchproject__(batch_index,cle_x_trainbatch, cle_y_trainbatch)       
                                cle_w_train.append(pro_w_trainbatch)                  
                                cle_y_train.append(pro_y_trainbatch)
                  

            else:
                raise Exception("参数 projected_dataset 不为空,无需投影！")

        cle_w_train_tensor = torch.stack(cle_w_train)                                                                         
        cle_y_train_tensor = torch.stack(cle_y_train)                                                                         

        self.cle_w_train = cle_w_train_tensor
        self.cle_y_train = cle_y_train_tensor

        print("self.cle_w_train.type:",type(self.cle_w_train))         
        print("self.cle_w_train.dtype:",self.cle_w_train.dtype)       
        print("self.cle_w_train.shape:",self.cle_w_train.shape)       

        print("self.cle_y_train.type:",type(self.cle_y_train))         
        print("self.cle_y_train.dtype:",self.cle_y_train.dtype)       
        print("self.cle_y_train.shape:",self.cle_y_train.shape)  

        print(f"Finished projecting {self._args.dataset} the whole {sample_num} samples!")


    def interpolatemain(self):

        mix_w_train, mix_y_train = self.interpolate()                                                                            
        self.mix_w_train = mix_w_train
        self.mix_y_train = mix_y_train

        print("self.mix_w_train.type:",type(self.mix_w_train))          
        print("self.mix_w_train.dtype:",self.mix_w_train.dtype)       
        print("self.mix_w_train.shape:",self.mix_w_train.shape)       

        print("self.mix_y_train.type:",type(self.mix_y_train))         
        print("self.mix_y_train.dtype:",self.mix_y_train.dtype)       
        print("self.mix_y_train.shape:",self.mix_y_train.shape)  
        
        print(f"Finished interpolate {self._args.dataset} {len(self.mix_w_train)} samples!")
        self.generatemain()
        print(f"Finished generate {self._args.dataset} {len(self.mix_w_train)} interpolated samples!")
        
    def generatemain(self):
        generated_x_train, generated_y_train = self.generate()
        self.generated_x_train = generated_x_train
        self.generated_y_train = generated_y_train
        print(f"Finished generate {self._args.dataset} {len(self.mix_w_train)} interpolated samples!")

    def mixgenerate(self,cle_train_dataloader) -> "tensor" :

        if self._args.mix_dataset == None:     
            mix_w_train, mix_y_train = self.interpolate()                                                                     
            self.mix_w_train = mix_w_train
            self.mix_y_train = mix_y_train

            generated_x_train, generated_y_train = self.generate()
            self.generated_x_train = generated_x_train
            self.generated_y_train = generated_y_train
        else:
            print(f"load mixed sampels from {self._args.mix_dataset}")

    def generatedset(self):
        if self._args.mix_dataset == None:
            return self.generated_x_train,self.generated_y_train
        else:           
            generated_x_train, generated_y_train = self.getmixset(self._args.mix_dataset)
            self.generated_x_train = generated_x_train
            self.generated_y_train = generated_y_train
            return self.generated_x_train,self.generated_y_train

    def getmixset(self,mix_dataset_path):
        
        file_dir=os.listdir(mix_dataset_path)
        file_dir.sort()

        img_filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[-9:-4] == 'image']           
        label_filenames = [name for name in file_dir if os.path.splitext(name)[-1] == '.npz' and name[-9:-4] == 'label']           
        
        select_mix_num = int( self._args.aug_mix_rate * self._args.aug_num )
        
        mix_xset_tensor = []
        for miximg_index, img_filename in enumerate(img_filenames):
            if miximg_index < select_mix_num:
                mix_img_npz_path = os.path.join(mix_dataset_path,img_filename)
                load_mix_img = np.load(mix_img_npz_path)['w']            
                load_mix_img = torch.tensor(load_mix_img)
                mix_xset_tensor.append(load_mix_img)

        mix_yset_tensor = []
        for mixy_index, lab_filename in enumerate(label_filenames):
            if mixy_index < select_mix_num:  
                mix_lab_npz_path = os.path.join(mix_dataset_path,lab_filename)
                load_mix_lab = np.load(mix_lab_npz_path)['w']            
                load_mix_lab = torch.tensor(load_mix_lab)
                mix_yset_tensor.append(load_mix_lab)

        mix_xset_tensor = torch.stack(mix_xset_tensor)                                                                         
        mix_yset_tensor = torch.stack(mix_yset_tensor)  

        return mix_xset_tensor.cuda(), mix_yset_tensor.cuda()

    def project(self):
        if self._args.gen_model == "stylegan2ada":

            if self._args.viewdataset_path == None:                                                 
                self._model.project(self._exp_result_dir,self.cle_x_train,self.cle_y_train)         
            elif self._args.viewdataset_path != None:
                self._model.project(self._exp_result_dir)

            cle_w_train, cle_y_train = self._model.wyset()                                                                    

            cle_w_train = torch.stack(cle_w_train)                                                                           
            print('pro_w_train.shape:',cle_w_train.shape)                                                                                  
            cle_y_train = torch.stack(cle_y_train)                                                                              
            print('pro_y_train.shape:',cle_y_train.shape)                                                                          
        return cle_w_train,cle_y_train  
                                                                                            
    def __batchproject__(self, batch_index, cle_x_trainbatch, cle_y_trainbatch):
        if self._args.gen_model == "stylegan2ada":
            self._model.project(self._exp_result_dir,cle_x_trainbatch,cle_y_trainbatch,batch_index)         
            cle_w_train, cle_y_train = self._model.wyset()                                                                     
            cle_w_train = torch.stack(cle_w_train)                                                                             
            print('pro_w_train.shape:',cle_w_train.shape)                                                                     
            cle_y_train = torch.stack(cle_y_train)                                                                              
            print('pro_y_train.shape:',cle_y_train.shape)                                                                       
        return cle_w_train,cle_y_train  

    def interpolate(self):
        if self._args.gen_model == "stylegan2ada":
            
            """
            self._args.projected_w1: None
            self.cle_w_train.shape: torch.Size([4, 8, 512])
            self.cle_y_train.shape: torch.Size([4, 8, 10])
            """
            if self._args.defense_mode == 'rmt':

                self._model.interpolate(self._exp_result_dir, self.cle_w_train, self.cle_y_train)
                mix_w_train, mix_y_train = self._model.mixwyset()  

            else:
                if self._args.projected_dataset is None and self._args.projected_w1 is None :       
                    self._model.interpolate(self._exp_result_dir, self.cle_w_train, self.cle_y_train)
                
                elif self._args.projected_dataset is not None or self._args.projected_w1 is not None:   
                    self._model.interpolate(self._exp_result_dir)                                                                  
                mix_w_train, mix_y_train = self._model.mixwyset()                                                                  
                mix_w_train = torch.stack(mix_w_train).cuda()                                                                            
                mix_y_train = torch.stack(mix_y_train).cuda()                                                                             
                """
                mix_w_train.shape: torch.Size([4, 8, 512])
                mix_y_train.shape: torch.Size([4, 8, 10])
                """

            return mix_w_train, mix_y_train                        

    def generate(self):
        if self._args.gen_model == "stylegan2ada":

            if self._args.defense_mode == 'rmt':

                self._model.generate(self._exp_result_dir, self.mix_w_train, self.mix_y_train)
                generated_x_train, generated_y_train = self._model.genxyset()                                                            
            else:

                if self._args.mixed_dataset ==None:
                    
                    if self._args.generate_seeds is not None:
                        self._model.generate(self._exp_result_dir)
                    else:
                        self._model.generate(self._exp_result_dir, self.mix_w_train, self.mix_y_train) 

                elif self._args.mixed_dataset !=None:
                    self._model.generate(self._exp_result_dir)

                generated_x_train, generated_y_train = self._model.genxyset() 
                generated_x_train = torch.stack(generated_x_train)                                                             
                generated_y_train = torch.stack(generated_y_train)                                                             
        return generated_x_train, generated_y_train    
    