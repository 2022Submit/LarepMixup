"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""


def SaveAccuracyTxt(pretrain,model,dataset,exp_result_dir,accuracy):
    if pretrain == True:
        accuracy_txt=open(f'{exp_result_dir}/classifier-accuracy-of-pretrained-{model}-on-imagenet.txt', "w")    
        txt_content = f'{exp_result_dir}/classifier-accuracy-of-pretrained-{model}-on-imagenet = {accuracy}'
        accuracy_txt.write(str(txt_content))
    elif pretrain == False:
        accuracy_txt=open(f'{exp_result_dir}/classifier-accuracy-of-selftrained-{model}-on-{dataset}.txt', "w")    
        txt_content = f'{exp_result_dir}/classifier-accuracy-of-selftrained-{model}-on-{dataset} = {accuracy}'
        accuracy_txt.write(str(txt_content))

def SaveLossTxt(pretrain,model,dataset,exp_result_dir,loss):
    if pretrain == True:
        loss_txt=open(f'{exp_result_dir}/classifier-loss-of-pretrained-{model}-on-imagenet.txt', "w")    
        loss_txt_content = f'{exp_result_dir}/classifier-loss-of-pretrained-{model}-on-imagenet = {loss}'
        loss_txt.write(str(loss_txt_content))
    elif pretrain == False:
        loss_txt=open(f'{exp_result_dir}/classifier-loss-of-selftrained-{model}-on-{dataset}.txt', "w")    
        loss_txt_content = f'{exp_result_dir}/classifier-loss-of-selftrained-{model}-on-{dataset} = {loss}'
        loss_txt.write(str(loss_txt_content))


   
def SaveTxt(args,exp_result_dir,
    cle_test_accuracy, 
    adv_test_accuracy, 
    trained_adv_test_accuracy,
    trained_cle_test_accuracy,

    cle_test_loss, 
    adv_test_loss, 
    trained_adv_test_loss,
    trained_cle_test_loss
    ):
    if args.defense_mode == "at":
        if args.pretrained_on_imagenet == True:
            accuracy_txt=open(f'{exp_result_dir}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
            txt_content = f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {cle_test_accuracy} \n'
            txt_content += f'{exp_result_dir}/adversarial-trained-classifier-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {trained_cle_test_accuracy}\n'
            txt_content += f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
            txt_content += f'{exp_result_dir}/adversarial-trained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {trained_adv_test_accuracy}\n'
            accuracy_txt.write(str(txt_content))
        
            loss_txt=open(f'{exp_result_dir}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-loss-on-clean-imagenet-testset = {cle_test_loss}\n'
            loss_txt_content += f'{exp_result_dir}/adversarial-trained-classifier-{args.cla_model}-loss-on-cle-{args.dataset}-testset = {trained_cle_test_loss}\n'
            loss_txt_content += f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'
            loss_txt_content += f'{exp_result_dir}/adversarial-trained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {trained_adv_test_loss}\n'
            loss_txt.write(str(loss_txt_content))    
        
        elif args.pretrained_on_imagenet == False:
            accuracy_txt=open(f'{exp_result_dir}/{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
            txt_content = f'{exp_result_dir}/standard-trained-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {cle_test_accuracy:08f}\n'
            txt_content += f'{exp_result_dir}/adversarial-trained-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {trained_cle_test_accuracy:08f}\n'
            txt_content += f'{exp_result_dir}/standard-trained-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy:08f}\n'
            txt_content += f'{exp_result_dir}/adversarial-trained-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {trained_adv_test_accuracy:08f}\n'
            accuracy_txt.write(str(txt_content))

            loss_txt=open(f'{exp_result_dir}/{args.cla_model}-loss-on-clean-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{exp_result_dir}/standard-trained-{args.cla_model}-loss-on-cle-{args.dataset}-testset = {cle_test_loss:08f}\n'
            loss_txt_content += f'{exp_result_dir}/adversarial-trained-{args.cla_model}-loss-on-cle-{args.dataset}-testset = {trained_cle_test_loss:08f}\n'
            loss_txt_content += f'{exp_result_dir}/standard-trained-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss:08f}\n'
            loss_txt_content += f'{exp_result_dir}/adversarial-trained-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {trained_adv_test_loss:08f}\n'
            loss_txt.write(str(loss_txt_content))
            
    elif args.defense_mode == "mmat":
        if args.pretrained_on_imagenet == True:
            accuracy_txt=open(f'{exp_result_dir}/classifier-{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
            txt_content = f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {cle_test_accuracy} \n'
            txt_content += f'{exp_result_dir}/manifold mixup adversarial-trained-classifier-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {trained_cle_test_accuracy}\n'
            txt_content += f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy}\n'
            txt_content += f'{exp_result_dir}/manifold mixup adversarial-trained-classifier-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {trained_adv_test_accuracy}\n'

            accuracy_txt.write(str(txt_content))
        
            loss_txt=open(f'{exp_result_dir}/classifier-{args.cla_model}-loss-on-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-loss-on-cle-imagenet-testset = {cle_test_loss}\n'
            loss_txt_content += f'{exp_result_dir}/manifold mixup adversarial-trained-classifier-{args.cla_model}-loss-on-cle-{args.dataset}-testset = {trained_cle_test_loss}\n'
            loss_txt_content += f'{exp_result_dir}/pretrained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss}\n'
            loss_txt_content += f'{exp_result_dir}/manifold mixup adversarial-trained-classifier-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {trained_adv_test_loss}\n'
            loss_txt.write(str(loss_txt_content))    
        
        elif args.pretrained_on_imagenet == False:   
            accuracy_txt=open(f'{exp_result_dir}/{args.cla_model}-accuracy-on-{args.dataset}-testset.txt', "w")    
            txt_content = f'{exp_result_dir}/standard-trained-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {cle_test_accuracy:08f}\n'
            txt_content += f'{exp_result_dir}/mmat-trained-{args.cla_model}-accuracy-on-cle-{args.dataset}-testset = {trained_cle_test_accuracy:08f}\n'
            txt_content += f'{exp_result_dir}/standard-trained-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {adv_test_accuracy:08f}\n'
            txt_content += f'{exp_result_dir}/mmat-trained-{args.cla_model}-accuracy-on-adv-{args.dataset}-testset = {trained_adv_test_accuracy:08f}\n'
            accuracy_txt.write(str(txt_content))

            loss_txt=open(f'{exp_result_dir}/{args.cla_model}-loss-on-clean-{args.dataset}-testset.txt', "w")    
            loss_txt_content = f'{exp_result_dir}/standard-trained-{args.cla_model}-loss-on-cle-{args.dataset}-testset = {cle_test_loss:08f}\n'
            loss_txt_content += f'{exp_result_dir}/mmat-trained-{args.cla_model}-loss-on-cle-{args.dataset}-testset = {trained_cle_test_loss:08f}\n'
            loss_txt_content += f'{exp_result_dir}/standard-trained-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {adv_test_loss:08f}\n'
            loss_txt_content += f'{exp_result_dir}/mmat-trained-{args.cla_model}-loss-on-adv-{args.dataset}-testset = {trained_adv_test_loss:08f}\n'

            loss_txt.write(str(loss_txt_content))