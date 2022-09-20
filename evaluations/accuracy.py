"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import torch
import torch.utils.data

def EvaluateAccuracy(classifier, classify_loss, test_dataloader:torch.utils.data.DataLoader, cla_model_name):
    classifier.eval()   
    testset_total_num = len(test_dataloader.dataset)
    epoch_correct_num = 0
    epoch_total_loss = 0
    
    for batch_index,(images, labels) in enumerate(test_dataloader):

        imgs = images.cuda()
        labs = labels.cuda()
        
        with torch.no_grad():

            if cla_model_name == 'inception_v3':
                output, aux = classifier(imgs)
            elif cla_model_name == 'googlenet':
                if images.size(-1) == 256:  
                    output = classifier(imgs)
                else:
                    output, aux1, aux2 = classifier(imgs)
            else:
                
                if images[0].shape == (3,256,256):                         
                    output = classifier(imgs, imagenetmixed10=True)
                else:
                    output = classifier(imgs)
            
            loss = classify_loss(output,labs)
            _, predicted_label_index = torch.max(output.data, 1)    

            batch_same_num = (predicted_label_index == labs).sum().item()
            epoch_correct_num += batch_same_num
            epoch_total_loss += loss

    test_accuracy = epoch_correct_num / testset_total_num
    test_loss = epoch_total_loss / len(test_dataloader)     

    return test_accuracy,test_loss 
    