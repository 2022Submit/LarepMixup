""" Author: maggie  Date:   2022-09-19  Place:  Xidian University   @copyright  """

from logging import error
import torch
import utils.parseargs
from clamodels.classifier import RMClassifier
from datas.dataset import RMDataset
from datas.dataloader import RMDataloader
from attacks.advattack import AdvAttack
from utils.savetxt import SaveTxt
from genmodels.mixgenerate import MixGenerate
import utils.stylegan2ada.dnnlib as dnnlib       
import utils.stylegan2ada.legacy as legacy
import os
import numpy as np
from attacks.perattack import PerAttack



if __name__ == '__main__':
    print("\n")
    print("---------------------------------------")

    if torch.cuda.is_available():
        print('Torch cuda is available')
    else:
        raise Exception('Torch cuda is not available')

    args, exp_result_dir, stylegan2ada_config_kwargs = utils.parseargs.main()
    
    cle_dataset = RMDataset(args)
    cle_train_dataset = cle_dataset.traindataset()
    cle_test_dataset = cle_dataset.testdataset()
    
    cle_dataloader = RMDataloader(args,cle_train_dataset,cle_test_dataset)
    cle_train_dataloader = cle_dataloader.traindataloader()
    cle_test_dataloader = cle_dataloader.testdataloader()

    print("cle_train_dataloader.len",len(cle_train_dataloader))
    print("cle_test_dataloader.len",len(cle_test_dataloader))

    """
    cle_train_dataloader.len 2414
    cle_test_dataloader.len 94
    """
    if args.mode == 'train':
        if args.train_mode =="gen-train":                                               
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
        
        elif args.train_mode =="cla-train":    
            target_classifier = RMClassifier(args)

            if args.pretrained_on_imagenet == False:
                target_classifier.train(cle_train_dataloader,cle_test_dataloader,exp_result_dir, train_mode = 'std-train')                                  
            elif args.pretrained_on_imagenet == True:
                target_classifier = target_classifier

            cle_test_accuracy, cle_test_loss = target_classifier.evaluatefromdataloader(target_classifier.model(),cle_test_dataloader)
            print(f'standard trained classifier *accuary* on clean testset:{cle_test_accuracy * 100:.4f}%' )                                    
            print(f'standard trained classifier *loss* on clean testset:{cle_test_loss}' ) 

    elif args.mode == 'attack':
        if args.latentattack == False:    
            if args.perceptualattack == False:  
                
                if args.attack_mode =='cw':     
                    print("confidence:",args.confidence)
                else:
                    print("eps:",args.attack_eps)                   

                print("pixel adversarial attack.............")
                print("cla_network_pkl:",args.cla_network_pkl)
                """
                cla_network_pkl: /root/autodl-tmp/maggie/result/train/cla-train/preactresnet18-imagenetmixed10/standard-trained-classifier-preactresnet18-on-clean-imagenetmixed10-epoch-0023-acc-90.47.pkl
                """

                learned_model = torch.load(args.cla_network_pkl)
                attack_classifier = AdvAttack(args,learned_model)
                target_model = attack_classifier.targetmodel()    

                print("start generating adv 20220812")
                x_test_adv, y_test_adv = attack_classifier.generate(exp_result_dir, test_dataloader=cle_train_dataloader)         

                adv_test_accuracy, adv_test_loss = attack_classifier.evaluatefromtensor(target_model,x_test_adv,y_test_adv)
                print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
                print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

                accuracy_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
                txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
                accuracy_txt.write(str(txt_content))
            
                loss_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
                loss_txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'
                loss_txt.write(str(loss_txt_content))    
            
            elif args.perceptualattack == True: 
                learned_model = torch.load(args.cla_network_pkl)
                attack_classifier = PerAttack(args,learned_model)
                target_model = attack_classifier.targetmodel()    

                x_test_per, y_test_per = attack_classifier.generate(exp_result_dir, cle_test_dataloader)          

                target_model.eval()
                per_test_accuracy, per_test_loss = attack_classifier.evaluatefromtensor(target_model, x_test_per, y_test_per)
                print(f'standard trained classifier accuary on perceptual attack testset:{per_test_accuracy * 100:.4f}%' ) 
                print(f'standard trained classifier loss on perceptual attack testset:{per_test_loss}' )    

                accuracy_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
                txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-accuracy-on-per-{args.dataset}-testset = {per_test_accuracy}\n'
                accuracy_txt.write(str(txt_content))
            
                loss_txt=open(f'{attack_classifier.getexpresultdir()}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
                loss_txt_content = f'{attack_classifier.getexpresultdir()}/pretrained-classifier-{args.cla_model}-loss-on-per-{args.dataset}-testset = {per_test_loss}\n'
                loss_txt.write(str(loss_txt_content))    
        
        elif args.latentattack == True: 
            print("eps:",args.attack_eps)
            print("latent adversarial attack.............")
            print("cla_network_pkl:",args.cla_network_pkl)
            learned_cla_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_cla_model)
            cle_w_test, cle_y_test = target_classifier.getproset(args.projected_dataset)
            cle_y_test = cle_y_test[:,0]    
            print("cle_w_test.shape:",cle_w_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)

            cla_net = learned_cla_model
            cla_net.cuda()
            cla_net.eval()                

            device = torch.device('cuda')
            with dnnlib.util.open_url(args.gen_network_pkl) as fp:
                G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)                                                
            
            gan_net = G.synthesis     
            gan_net.cuda()
            gan_net.eval()

            merge_model = torch.nn.Sequential(gan_net, cla_net)
            latent_attacker = AdvAttack(args,merge_model)
            
            adv_x_test, adv_y_test = latent_attacker.generatelatentadv(exp_result_dir, cle_test_dataloader, cle_w_test, cle_y_test, gan_net)             
            adv_test_accuracy, adv_test_loss = latent_attacker.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
            print(f'standard trained classifier accuary on adversarial testset:{adv_test_accuracy * 100:.4f}%' ) 
            print(f'standard trained classifier loss on adversarial testset:{adv_test_loss}' )    

        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")    
                    
    elif args.mode == 'project':        
        if args.gen_network_pkl != None:        
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)

            print("projecting test set-----为了避免在跑代码的中断，文件夹仍然名为trainset")
            generate_model.projectmain(cle_test_dataloader)     

        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")

    elif args.mode == 'interpolate':
        if args.gen_network_pkl != None:     
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)
            generate_model.interpolatemain() 
        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")    

    elif args.mode == 'generate':
        if args.gen_network_pkl != None:     
            generate_model = MixGenerate(args, exp_result_dir, stylegan2ada_config_kwargs)                        
            generate_model.generatemain()
        else:
            raise Exception("There is no gen_network_pkl, please train generative model first!")    

    elif args.mode == 'defense':        
        if args.defense_mode == "rmt":
            print("adversarial training")
            print("args.attack_mode:",args.attack_mode)
            print("lr:",args.lr)

            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            print("args.projected_dataset",args.projected_dataset)
            cle_w_train, cle_y_train = target_classifier.getproset(args.projected_dataset)
            print("cle_w_train.shape:",cle_w_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)

            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)

            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
            
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     
            print(f'Accuary of before rmt trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before rmt trained classifier clean testset:{cle_test_loss}' ) 

            if args.cla_model in ['preactresnet18','preactresnet34','preactresnet50'] and args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                raise error

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_mode == "om-pgd":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                raise error

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_eps != 0.02:    
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                raise error

            if args.cla_model in ['alexnet','resnet18','resnet34','resnet50','vgg19','densenet169','googlenet'] and args.attack_mode in ["fog","snow","elastic","jpeg"]:
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before rmt trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before rmt trained classifier on adv testset:{adv_test_loss}' ) 
                raise error            

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)
            
            target_classifier.rmt(args,cle_w_train,cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir,stylegan2ada_config_kwargs)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of rmt trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of rmt trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of rmt trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of rmt trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of rmt trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of rmt trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='at':
            print("adversarial training")
            print("args.attack_mode:",args.attack_mode)
            print("lr:",args.lr)

            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)

            adv_trainset_path = args.train_adv_dataset

            adv_x_train, adv_y_train = target_classifier.getadvset(adv_trainset_path)
            print("adv_x_train.shape:",adv_x_train.shape)
            print("adv_y_train.shape:",adv_y_train.shape)            

            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset

            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
            print("adv_x_test.shape:",adv_x_test.shape)
            print("adv_y_test.shape:",adv_y_test.shape)  

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     
            print(f'Accuary of before adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before adversarial trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before at trained classifier on adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before at trained classifier on adv testset:{adv_test_loss}' ) 
                raise error
            
            target_classifier.advtrain(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of adversarial trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of adversarial trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of adversarial trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of adversarial trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of adversarial trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='inputmixup':
            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()
            print("cle_y_train.shape:",cle_y_train.shape)
            
            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     
            print(f'Accuary of before inputmixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before inputmixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before inputmixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before inputmixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.inputmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of inputmixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of inputmixup trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of inputmixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of inputmixup trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of inputmixup trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of inputmixup trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='manifoldmixup':
            print("manifold mixup")
            print("lr:",args.lr)
            print("cla_network_pkl:",args.cla_network_pkl)
            print("args.attack_mode:",args.attack_mode)

            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  
            print("cle_y_train.shape:",cle_y_train.shape)

            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     
            print(f'Accuary of before manifold mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before manifold mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)   
                print(f'Accuary of before manifold mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before manifold mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.manifoldmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)


            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of manifold trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of manifold trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of manifold trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of manifold trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of manifold trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of manifold trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='patchmixup':
            print("patch mixup")
            print("lr:",args.lr)
            print("cla_network_pkl:",args.cla_network_pkl)
            print("args.attack_mode:",args.attack_mode)

            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  
            print("cle_y_train.shape:",cle_y_train.shape)

            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     
            print(f'Accuary of before patch mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before patch mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)     
                print(f'Accuary of before patch mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before patch mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.patchmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)


            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of patch trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of patch trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of patch trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of patch trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of patch trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of patch trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='puzzlemixup':
            print("puzzle mixup")
            print("lr:",args.lr)
            print("args.attack_mode:",args.attack_mode)
            print("cla_network_pkl:",args.cla_network_pkl)

            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float()  
            print("cle_y_train.shape:",cle_y_train.shape)

            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)     
            print(f'Accuary of before puzzle mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before puzzle mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)    
                print(f'Accuary of before puzzle mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before puzzle mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.puzzlemixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of puzzle trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of puzzle trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of puzzle trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of puzzle trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of puzzle trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of puzzle trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='cutmixup':
            print("cut mixup")
            print("lr:",args.lr)
            print("cla_network_pkl:",args.cla_network_pkl)
            print("args.attack_mode:",args.attack_mode)

            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)
            cle_y_train = torch.nn.functional.one_hot(cle_y_train, args.n_classes).float() 
            print("cle_y_train.shape:",cle_y_train.shape)

            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset
            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)    
            print(f'Accuary of before cut mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before cut mixup trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)    
                print(f'Accuary of before cut mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before cut mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 
                raise error

            print("args.mix_mode:",args.mix_mode)
            print("args.mix_w_num:",args.mix_w_num)
            print("args.beta_alpha:",args.beta_alpha)
            print("args.dirichlet_gama:",args.dirichlet_gama)

            target_classifier.cutmixuptrain(args, cle_x_train, cle_y_train, cle_train_dataloader, cle_x_test,cle_y_test,adv_x_test,adv_y_test,exp_result_dir)

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of cut mixup trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of cut mixup trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of cut mixup trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of cut mixup trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of cut mixup trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of cut mixup trained classifier on black-box adv testset:{adv_test_loss}' ) 

        elif args.defense_mode =='dmat':
            print("dual manifold adversarial training")
            print("args.attack_mode:",args.attack_mode)
            print("lr:",args.lr)

            print("args.cla_network_pkl",args.cla_network_pkl)
            learned_model = torch.load(args.cla_network_pkl)
            target_classifier = RMClassifier(args,learned_model)

            cle_x_train, cle_y_train = target_classifier.getrawset(cle_train_dataloader)
            cle_x_test, cle_y_test = target_classifier.getrawset(cle_test_dataloader)
            print("cle_x_train.shape:",cle_x_train.shape)
            print("cle_y_train.shape:",cle_y_train.shape)
            print("cle_x_test.shape:",cle_x_test.shape)
            print("cle_y_test.shape:",cle_y_test.shape)

            print("args.train_adv_dataset",args.train_adv_dataset)
            adv_trainset_path = args.train_adv_dataset
            adv_x_train, adv_y_train = target_classifier.getadvset(adv_trainset_path)
            adv_x_train=adv_x_train[:25000]
            adv_y_train=adv_y_train[:25000]
            print("adv_x_train.shape:",adv_x_train.shape)
            print("adv_y_train.shape:",adv_y_train.shape)            

            print("args.train_adv_dataset_2",args.train_adv_dataset_2)
            adv_trainset_path_2 = args.train_adv_dataset_2
            adv_x_train_2, adv_y_train_2 = target_classifier.getadvset(adv_trainset_path_2)
            adv_x_train_2=adv_x_train_2[:25000]
            adv_y_train_2=adv_y_train_2[:25000]            
            print("adv_x_train_2.shape:",adv_x_train_2.shape)
            print("adv_y_train_2.shape:",adv_y_train_2.shape)  
              
            adv_x_train = torch.cat([adv_x_train, adv_x_train_2], dim=0)
            adv_y_train = torch.cat([adv_y_train, adv_y_train_2], dim=0)  
            print("adv_x_train.shape:",adv_x_train.shape)
            print("adv_y_train.shape:",adv_y_train.shape)  

            print("args.test_adv_dataset",args.test_adv_dataset)
            adv_testset_path = args.test_adv_dataset

            adv_x_test, adv_y_test = target_classifier.getadvset(adv_testset_path)
            print("adv_x_test.shape:",adv_x_test.shape)
            print("adv_y_test.shape:",adv_y_test.shape)  

            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)    
            print(f'Accuary of before dual manifold adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of before dual manifold adversarial trained classifier clean testset:{cle_test_loss}' ) 

            if args.attack_mode != "fgsm":
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of before dual manifold adversarial trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of before dual manifold adversarial trained classifier on white-box adv testset:{adv_test_loss}' )           
                raise error("maggie stop here")
            
            target_classifier.advtrain(args, cle_train_dataloader, adv_x_train, adv_y_train, cle_x_test, cle_y_test, adv_x_test, adv_y_test, exp_result_dir)
            cle_test_acc, cle_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),cle_x_test,cle_y_test)
            print(f'Accuary of dual manifold adversarial trained classifier on clean testset:{cle_test_acc * 100:.4f}%' ) 
            print(f'Loss of dual manifold adversarial trained classifier on clean testset:{cle_test_loss}' ) 
           
            if args.whitebox == True:
                attack_classifier = AdvAttack(args, target_classifier.model())
                target_model = attack_classifier.targetmodel()
                adv_x_test, adv_y_test = attack_classifier.generateadvfromtestsettensor(cle_x_test, cle_y_test)
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_model,adv_x_test,adv_y_test)
                print(f'Accuary of dual manifold adversarial trained classifier on white-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of dual manifold adversarial trained classifier on white-box adv testset:{adv_test_loss}' ) 

            elif args.blackbox == True:
                adv_x_test, adv_y_test = adv_x_test, adv_y_test
                adv_test_acc, adv_test_loss = target_classifier.evaluatefromtensor(target_classifier.model(),adv_x_test,adv_y_test)
                print(f'Accuary of dual manifold adversarial trained classifier on black-box adv testset:{adv_test_acc * 100:.4f}%' ) 
                print(f'Loss of dual manifold adversarial trained classifier on black-box adv testset:{adv_test_loss}' ) 
        
    print("---------------------------------------")
    print("\n")