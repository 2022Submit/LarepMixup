
from logging import error
import torch
from art.estimators.classification import PyTorchClassifier
from torch.cuda import device
import torchvision
import numpy as np
import art.attacks.evasion
from utils.savepng import save_image
import copy
import os
from torch import LongTensor
import advertorch.attacks

Tensor = torch.Tensor

class AdvAttack():
    r"""
        class of the adversarial attack 
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
        print('initlize attack classifier')
        if args.latentattack == False:   
            print("generate pixel adversarial exampels")
            self._args = args
            self._targetmodel = learned_model         
            if self._args.whitebox == True:
                self._model = copy.deepcopy(learned_model)
            elif self._args.blackbox == True:
                self._model = torchvision.models.resnet34(pretrained=True)

            self._lossfunc = self.__getlossfunc__()
            self._optimizer = self.__getoptimizer__()

            self._artmodel = self.__getartmodel__()

            self._advgenmodel = self.__getadvgenmodel__()

        elif args.latentattack == True:
            print("generate latent adversarial exampels")
            self._args = args
            self._targetmodel = learned_model  
            if self._args.whitebox == True:
                self._model = copy.deepcopy(learned_model)
            elif self._args.blackbox == True:
                self._model = torchvision.models.resnet34(pretrained=True)

            self._attacker = self.__getadvertorchmodel__() 

    def __getadvertorchmodel__(self):  
        if self._args.attack_mode =='pgd':
            print("latent pgd attack")
            attacker = advertorch.attacks.PGDAttack(predict=self._model, eps=self._args.attack_eps, eps_iter=0.005, nb_iter=100, clip_min=None, clip_max=None)      
            
        elif self._args.attack_mode =='fgsm':
            print("latent fgsm attack")
            print("eps：",self._args.attack_eps)
            attacker = advertorch.attacks.GradientSignAttack(predict=self._model,eps=self._args.attack_eps,clip_min=None, clip_max=None)

        elif self._args.attack_mode =='cw':
            attacker = advertorch.attacks.CarliniWagnerL2Attack(predict=self._model,eps=self._args.attack_eps,clip_min=None, clip_max=None)

        return attacker

    def __getlossfunc__(self):
        # torch.nn.L1Loss
        # torch.nn.KLDivLoss
        # torch.nn.SmoothL1Loss
        # torch.nn.SoftMarginLoss
        # torch.nn.LocalResponseNorm
        # torch.nn.MultiMarginLoss
        # torch.nn.CrossEntropyLoss
        # torch.nn.BCEWithLogitsLoss
        # torch.nn.MarginRankingLoss
        # torch.nn.TripletMarginLoss
        # torch.nn.HingeEmbeddingLoss
        # torch.nn.CosineEmbeddingLoss
        # torch.nn.MultiLabelMarginLoss
        # torch.nn.MultiLabelSoftMarginLoss
        # torch.nn.AdaptiveLogSoftmaxWithLoss
        # torch.nn.TripletMarginWithDistanceLoss
        lossfunc = torch.nn.CrossEntropyLoss()
        return lossfunc
    
    def __getoptimizer__(self):
        # torch.optim.Adadelta()
        # torch.optim.Adagrad()
        # torch.optim.Adam()
        # torch.optim.Adamax()
        # torch.optim.AdamW()
        # torch.optim.ASGD()
        # torch.optim.LBFGS()
        # torch.optim.RMSprop()
        # torch.optim.Rprop()
        # torch.optim.SGD()
        # torch.optim.SparseAdam()
        # torch.optim.Optimizer()
        optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._args.lr)
        return optimizer

    def targetmodel(self):
        return self._targetmodel

    def __getartmodel__(self) -> "PyTorchClassifier":

        if torch.cuda.is_available():
            self._lossfunc.cuda()
            self._model.cuda()      
        
        data_raw = False                                      
        if data_raw == True:
            min_pixel_value = 0.0
            max_pixel_value = 255.0
        else:
            min_pixel_value = 0.0
            max_pixel_value = 1.0        

        artmodel = PyTorchClassifier(
            model=self._model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=self._lossfunc,
            optimizer=self._optimizer,
            input_shape=(self._args.channels, self._args.img_size, self._args.img_size),
            nb_classes=self._args.n_classes,
        )             

        return artmodel

    def __getadvgenmodel__(self) -> "art.attacks.evasion":
        
        if self._args.attack_mode == 'fgsm':                              
            print('Get FGSM examples generate model')
            print("self._args.attack_eps:",self._args.attack_eps)
            advgenmodel = art.attacks.evasion.FastGradientMethod(estimator=self._artmodel, eps=self._args.attack_eps, targeted=False)    

        elif self._args.attack_mode =='deepfool':                         
            print('Get DeepFool examples generate model')
            advgenmodel = art.attacks.evasion.DeepFool(classifier=self._artmodel, epsilon=self._args.attack_eps)                         

        elif self._args.attack_mode =='bim':                              
            print('Get BIM(PGD) examples generate model')
            advgenmodel = art.attacks.evasion.BasicIterativeMethod(estimator=self._artmodel, eps=self._args.attack_eps, targeted=False)  

        elif self._args.attack_mode =='cw':                               
            print('Get CW examples generate model')
            advgenmodel = art.attacks.evasion.CarliniL2Method(classifier=self._artmodel, confidence=self._args.confidence, targeted=False)               
        
        elif self._args.attack_mode =='pgd': 
            advgenmodel = art.attacks.evasion.ProjectedGradientDescent(estimator=self._artmodel, eps=self._args.attack_eps, targeted=False)   

        elif self._args.attack_mode =='autoattack':   
            advgenmodel = art.attacks.evasion.AutoAttack(estimator=self._artmodel, eps=self._args.attack_eps, estimator_orig=self._artmodel, targeted=False)

        elif self._args.attack_mode == None:
            raise Exception('please input the attack mode')      

        return advgenmodel

    def getexpresultdir(self):
        return self._exp_result_dir
    
    def generate(self, exp_result_dir, test_dataloader, train_dataloader = None) -> "Tensor":
        if train_dataloader is not None:
            self._train_dataloader = train_dataloader
            self._test_dataloader = test_dataloader 
            self._exp_result_dir = exp_result_dir

            self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
            os.makedirs(self._exp_result_dir,exist_ok=True)            

            self._x_train_adv=None
            self._y_train_adv=None

            self._x_train, self._y_train = self.__getsettensor__(self._train_dataloader)       
            print("finish get train list")
            self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)
            print("finish get test list")

            print("self._x_train.shape:",self._x_train.shape)
            print("self._y_train.shape:",self._y_train.shape)
            print("self._x_test.shape:",self._x_test.shape)
            print("self._y_test.shape:",self._y_test.shape)

            self._x_train = self._x_train.cpu().numpy()                            
            self._y_train = self._y_train.cpu().numpy()
            self._x_test = self._x_test.cpu().numpy()
            self._y_test = self._y_test.cpu().numpy()

            print('generating trainset adversarial examples...')
            self._x_train_adv = self._advgenmodel.generate(x = self._x_train, y = self._y_train)
            self._y_train_adv = self._y_train
            print('finished generate trainset adversarial examples !')
            print('generating testset adversarial examples...')
            self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
            self._y_test_adv = self._y_test
            print('finished generate testset adversarial examples !')

            print("self._x_train_adv.shape:",self._x_train_adv.shape)
            print("self._y_train_adv.shape:",self._y_train_adv.shape)
            print("self._x_test_adv.shape:",self._x_test_adv.shape)
            print("self._y_test_adv.shape:",self._y_test_adv.shape)

            self._x_train_adv = torch.from_numpy(self._x_train_adv).cuda()
            self._y_train_adv = torch.from_numpy(self._y_train_adv).cuda()
            self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
            self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

            self._x_train = torch.from_numpy(self._x_train).cuda()
            self._y_train = torch.from_numpy(self._y_train).cuda()
            self._x_test = torch.from_numpy(self._x_test).cuda()
            self._y_test = torch.from_numpy(self._y_test).cuda()

            self.__saveadvpng__()

            return self._x_train_adv, self._y_train_adv, self._x_test_adv, self._y_test_adv         

        elif train_dataloader is None:
            self._test_dataloader = test_dataloader 
            self._exp_result_dir = exp_result_dir

            self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
            os.makedirs(self._exp_result_dir,exist_ok=True)            

            self._x_test, self._y_test = self.__getsettensor__(self._test_dataloader)

            self._x_test = self._x_test.cpu().numpy()
            self._y_test = self._y_test.cpu().numpy()

            self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
            self._y_test_adv = self._y_test

            self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
            self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

            self._x_test = torch.from_numpy(self._x_test).cuda()
            self._y_test = torch.from_numpy(self._y_test).cuda()

            self.__saveadvpng__()
            return self._x_test_adv, self._y_test_adv    

    def generatelatentadv(self,exp_result_dir, cle_test_dataloader, cle_w_test, cle_y_test, gan_net):
        self._exp_result_dir = exp_result_dir
        self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
        os.makedirs(self._exp_result_dir,exist_ok=True)     
        
        cle_w_test = cle_w_test.cuda()
        cle_y_test = cle_y_test.cuda()  

        print("cle_w_test.shape:",cle_w_test.shape)                
        print("cle_y_test.shape:",cle_y_test.shape)  

        testset_total_num = int(cle_w_test.size(0))
        batch_size = self._args.batch_size 
        batch_num = int( np.ceil( int(testset_total_num) / float(batch_size) ) )
        print("testset_total_num:",testset_total_num)
        print("batch_size:",batch_size)
        print("batch_num:",batch_num)
        
        adv_x_test = []
        for batch_index in range(batch_num):                                       
            print("batch_index:",batch_index)
            cle_w_batch = cle_w_test[batch_index * batch_size : (batch_index + 1) * batch_size]
            cle_y_batch = cle_y_test[batch_index * batch_size : (batch_index + 1) * batch_size] 
            adv_w_batch = self._attacker.perturb(cle_w_batch, cle_y_batch)
            adv_x_batch = gan_net(adv_w_batch)            
            adv_x_test.append(adv_x_batch)

        adv_x_test = torch.cat(adv_x_test, dim=0)        
        adv_y_test = cle_y_test
        print("adv_x_test.shape:",adv_x_test.shape)            
        print("adv_y_test.shape:",adv_y_test.shape)

        self._x_test_adv = adv_x_test
        self._y_test_adv = adv_y_test
        print('finished generate latent adversarial examples !')

        self.__saveadvpng__()

        return adv_x_test, adv_y_test

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

    def __saveadvpng__(self):

        if self._args.latentattack == False:   

            classification = self.__labelnames__() 
            print("label_names:",classification)        
            
            os.makedirs(f'{self._exp_result_dir}/samples/train/',exist_ok=True)    
            os.makedirs(f'{self._exp_result_dir}/samples/test/',exist_ok=True)    

            if self._args.dataset != "imagenetmixed10":
                print(f"Saving {self._args.dataset} trainset adversarial examples...")
                for img_index, _ in enumerate(self._x_train_adv):
                    save_adv_img = self._x_train_adv[img_index]
                    img_true_label = self._y_train_adv[img_index]
                
                    np.savez(f'{self._exp_result_dir}/samples/train/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_adv_img.cpu().numpy())      
             
            print(f"Saving {self._args.dataset} testset adversarial examples...")            
            for img_index, _ in enumerate(self._x_test_adv):
                save_adv_img = self._x_test_adv[img_index]
                img_true_label = self._y_test_adv[img_index]

                np.savez(f'{self._exp_result_dir}/samples/test/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_adv_img.cpu().numpy())      

        elif self._args.latentattack == True: 
            classification = self.__labelnames__() 
            print("label_names:",classification)  

            os.makedirs(f'{self._exp_result_dir}/latent-attack-samples/train/',exist_ok=True)    
            os.makedirs(f'{self._exp_result_dir}/latent-attack-samples/test/',exist_ok=True)    

            print(f"Saving {self._args.dataset} trainset adversarial examples...")
            for img_index, _ in enumerate(self._x_test_adv):
                save_adv_img = self._x_test_adv[img_index]
                img_true_label = self._y_test_adv[img_index]

                np.savez(f'{self._exp_result_dir}/latent-attack-samples/train/{img_index:08d}-adv-{img_true_label}-{classification[int(img_true_label)]}.npz', w=save_adv_img.cpu().numpy())    
            
        print("save adversarial examples finished")

    def generateadvfromtestsettensor(self, testset_tensor_x, testset_tensor_y, exp_result_dir = None):
        if exp_result_dir is not None:
            self._exp_result_dir = exp_result_dir
            self._exp_result_dir = os.path.join(self._exp_result_dir,f'attack-{self._args.dataset}-dataset')
            os.makedirs(self._exp_result_dir,exist_ok=True)            

        self._x_test = testset_tensor_x
        self._y_test = testset_tensor_y

        self._x_test = self._x_test.cpu().numpy()
        self._y_test = self._y_test.cpu().numpy()

        print('generating testset adversarial examples...')
        self._x_test_adv = self._advgenmodel.generate(x = self._x_test, y = self._y_test)
        self._y_test_adv = self._y_test
        print('finished generate testset adversarial examples !')

        self._x_test_adv = torch.from_numpy(self._x_test_adv).cuda()
        self._y_test_adv = torch.from_numpy(self._y_test_adv).cuda()

        self._x_test = torch.from_numpy(self._x_test).cuda()
        self._y_test = torch.from_numpy(self._y_test).cuda()

        return self._x_test_adv, self._y_test_adv          

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
                    if self._args.dataset == 'imagenetmixed10' or self._args.dataset == 'kmnist' or self._args.dataset == 'cifar10' or self._args.dataset == 'svhn':  
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

        if self._args.dataset == 'imagenetmixed10':
            print("len(dataloader.dataset):",len(dataloader.dataset))  
            xset_tensor = []
            
            if len(dataloader.dataset) == 77237:                           
                jieduan_num = 15000
                for img_index in range(jieduan_num):
                    if img_index % 100 == 0: 
                        print("trainset_img_index:",img_index)
                    xset_tensor.append(dataloader.dataset[img_index][0])
            elif len(dataloader.dataset) == 3000:                           
                for img_index in range(len(dataloader.dataset)):
                    if img_index % 100 == 0: 
                        print("testset_img_index:",img_index)
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

        if self._args.dataset == 'imagenetmixed10':
            print("len(dataloader.dataset):",len(dataloader.dataset))
            yset_tensor = []

            if len(dataloader.dataset) == 77237:                      
                jieduan_num = 15000
                for img_index in range(jieduan_num):            
                    if img_index % 100 == 0: 
                        print("trainset_label_index:",img_index)                                
                    yset_tensor.append(dataloader.dataset[img_index][1])
            elif len(dataloader.dataset) == 3000:                          
                for img_index in range(len(dataloader.dataset)):
                    if img_index % 100 == 0: 
                        print("testset_label_index:",img_index)                                
                    yset_tensor.append(dataloader.dataset[img_index][1])

            yset_tensor = LongTensor(yset_tensor)                          

        return yset_tensor.cuda()      
