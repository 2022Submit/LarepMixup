import attacks.fog
import attacks.snow
import attacks.elastic
import attacks.gabor
import attacks.jpeg
import copy
import os
import torch
import numpy as np
from utils.savepng import save_image
from torch import LongTensor

Tensor = torch.Tensor


class PerAttack():
    r"""
        class of the perceptual attack 
        attributes:
        self._args
        self._model
        self._loss
        self._optimizer
        self._targetmodel
        self._test_dataloader
        self._whitebox
        self._artmodel
        self._advgenmodel

        methods:
        self.__init__()
        self.__getartmodel__()
        self.__getadvgenmodel__()
   
    """
    def __init__(self, args, learned_model) -> None:                 
        print('initlize perceptual classifier')
        self._args = args
        self._targetmodel = learned_model  
        self._model = copy.deepcopy(learned_model)
        self._attacker = self.__getperattackmodel__()

    def targetmodel(self):
        return self._targetmodel

    def __getperattackmodel__(self):  
        self._model.cuda()
        self._model.eval()   
        if self._args.attack_mode =='fog':
            print("pixel fog attack")
            attacker = attacks.fog.FogAttack(predict=self._model, nb_iters=200 , eps_max =128, step_size =0.002236, resolution =self._args.img_size )

        elif self._args.attack_mode =='snow':
            print("pixel snow attack")  
            attacker = attacks.snow.SnowAttack(predict=self._model, nb_iters=200 , eps_max =0.0625, step_size =0.002236, resolution =self._args.img_size )

        elif self._args.attack_mode =='elastic':
            print("pixel elastic attack") 
            attacker = attacks.elastic.ElasticAttack(predict=self._model, nb_iters=200 , eps_max =0.5, step_size =0.035355339059327376, resolution =self._args.img_size )

        elif self._args.attack_mode =='gabor':
            print("pixel gabor attack")  
            attacker = attacks.gabor.GaborAttack(predict=self._model, nb_iters=200 , eps_max =12.5, step_size =0.002236, resolution =self._args.img_size )

        elif self._args.attack_mode =='jpeg':
            print("pixel jpeg attack")   
            attacker = attacks.jpeg.JPEGAttack(predict=self._model, nb_iters=200 , eps_max =32, step_size =72.40773439350247, opt ='l1', resolution =self._args.img_size )  

        return attacker
    
    def generate(self,exp_result_dir, cle_test_dataloader):
        self._test_dataloader = cle_test_dataloader 
        self._exp_result_dir = exp_result_dir

        self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
        os.makedirs(self._exp_result_dir,exist_ok=True)            

        print('generating testset perceptual attack examples...')
        per_x_test = []
        per_y_test = []
        for batch_index, (images, labels) in enumerate(self._test_dataloader):
            print(f"batch_index = {batch_index:04d}/{len(self._test_dataloader):04d}")
            images = images.cuda()
            labels = labels.cuda()
            per_x_batch = self._attacker.perturb(images,labels)
            per_y_batch = labels
            per_x_test.append(per_x_batch)
            per_y_test.append(per_y_batch)

        per_x_test = torch.cat(per_x_test, dim=0) 
        per_y_test = torch.cat(per_y_test, dim=0)    

        print("per_x_test.shape:",per_x_test.shape)             
        print("per_y_test.shape:",per_y_test.shape)
        print("per_y_test[:10]:",per_y_test[:10]) 

        """
        per_x_test.shape: torch.Size([64, 3, 32, 32])
        per_y_test.shape: torch.Size([64])
        per_y_test[:10]: tensor([3, 8, 8, 0, 6, 6, 1, 6, 3, 1], device='cuda:0')        
        """

        self._x_test_per = per_x_test
        self._y_test_per = per_y_test
        self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)

        print("self._x_test.shape:",self._x_test.shape)             
        print("self._y_test.shape:",self._y_test.shape)
        print("self._y_test[:10]:",self._y_test[:10]) 
        """
        self._x_test.shape: torch.Size([10000, 3, 32, 32])
        self._y_test.shape: torch.Size([10000])
        self._y_test[:10]: tensor([3, 8, 8, 0, 6, 6, 1, 6, 3, 1], device='cuda:0')
        """

        self.__saveperpng__()

        return self._x_test_per, self._y_test_per        


    def __labelnames__(self):
        opt = self._args
        
        label_names = []
        
        if opt.dataset == 'cifar10':
            label_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

        elif opt.dataset == 'cifar100': 
            label_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        
        elif opt.dataset =='svhn':
            label_names = ['0','1','2','3','4','5','6','7','8','9']

        elif opt.dataset =='kmnist':
            label_names = ['0','1','2','3','4','5','6','7','8','9']
        
        elif opt.dataset =='stl10': 
            label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
        
        elif opt.dataset =='imagenetmixed10':
            label_names = ['dog,','bird','insect','monkey','car','feline','truck','fruit','fungus','boat']        
        else:
            raise Exception(" label name get wrong")            
        
        return label_names

    def __saveperpng__(self):
        if self._args.latentattack == False:      
            classification = self.__labelnames__() 
            os.makedirs(f'{self._exp_result_dir}/samples/test/',exist_ok=True)    
            print(f"Saving {self._args.dataset} testset perceptual attack examples...")
            for img_index, _ in enumerate(self._x_test_per):
                save_per_img = self._x_test_per[img_index]
                save_cle_img = self._x_test[img_index]
                img_true_label = self._y_test_per[img_index]
                np.savez(f'{self._exp_result_dir}/samples/test/{img_index:08d}-per-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_per_img.cpu().detach().numpy())      
                 

    def __getsettensor__(self,dataloader)->"Tensor":

        xset_tensor  = self.__getxsettensor__(dataloader)
        yset_tensor = self.__getysettensor__(dataloader)

        return xset_tensor, yset_tensor
    
    def __getxsettensor__(self,dataloader)->"Tensor":
        if self._args.dataset == 'cifar10':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                         

        elif self._args.dataset == 'cifar100':
            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
                                        

        elif self._args.dataset == 'imagenet':
            jieduan_num = 1000

            xset_tensor = []
            for img_index in range(jieduan_num):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)                                                                          
            
        elif self._args.dataset == 'svhn':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)   

        elif self._args.dataset == 'kmnist':

            xset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                xset_tensor.append(dataloader.dataset[img_index][0])
            xset_tensor = torch.stack(xset_tensor)  

        return xset_tensor.cuda()                                      

    def __getysettensor__(self,dataloader)->"Tensor":

        if self._args.dataset == 'cifar10':
            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                         

        elif self._args.dataset == 'cifar100':

            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                           
           

        elif self._args.dataset == 'imagenet':
            jieduan_num = 1000   
            yset_tensor = []
            for img_index in range(jieduan_num):
                yset_tensor.append(dataloader.dataset[img_index][1])
            yset_tensor = LongTensor(yset_tensor)                           

        elif self._args.dataset == 'svhn':
            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])
            yset_tensor = LongTensor(yset_tensor)           

        elif self._args.dataset == 'kmnist':
            yset_tensor = []
            for img_index in range(len(dataloader.dataset)):
                yset_tensor.append(dataloader.dataset[img_index][1])
            yset_tensor = LongTensor(yset_tensor)                       
        return yset_tensor.cuda()       

    def evaluatefromtensor(self, classifier, x_set:Tensor, y_set:Tensor):
        classifier.eval()  
        if torch.cuda.is_available():
            classifier.cuda()             
        
        batch_size = self._args.batch_size
        testset_total_num = len(x_set)
        batch_num = int( np.ceil( int(testset_total_num) / float(batch_size) ) )
        cla_model_name=self._args.cla_model

        eva_loss = torch.nn.CrossEntropyLoss()
        epoch_correct_num = 0
        epoch_total_loss = 0

        for batch_index in range(batch_num):                                               
            images = x_set[batch_index * batch_size : (batch_index + 1) * batch_size]
            labels = y_set[batch_index * batch_size : (batch_index + 1) * batch_size]                                                

            imgs = images.cuda()
            labs = labels.cuda()

            with torch.no_grad():

                if cla_model_name == 'inception_v3':
                    output, aux = classifier(imgs)
                
                elif cla_model_name == 'googlenet':
                    if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'svhn' or self._args.dataset == 'kmnist' or self._args.dataset == 'cifar10':              
                        output = classifier(imgs)
                    else:
                        output, aux1, aux2 = classifier(imgs)
                else:
                    output = classifier(imgs)         
                                
                loss = eva_loss(output,labs)
                _, predicted_label_index = torch.max(output.data, 1)  
                
                batch_same_num = (predicted_label_index == labs).sum().item()
                epoch_correct_num += batch_same_num
                epoch_total_loss += loss


        test_accuracy = epoch_correct_num / testset_total_num
        test_loss = epoch_total_loss / batch_num                  
        classifier.train()

        return test_accuracy, test_loss

    def getexpresultdir(self):
        return self._exp_result_dir
    
