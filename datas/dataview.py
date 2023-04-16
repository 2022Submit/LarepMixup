

import os
import torch.utils.data
import PIL.Image
import utils.util


def ViewMNISTTrain(target_dataset):
    data = target_dataset              
    assert data is not None
    assert isinstance(data, str)
    print('data=%s'%data)

    training_set_kwargs = utils.util.EasyDict(class_name='utils.training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)  
    training_set = utils.util.construct_class_by_name(**training_set_kwargs) 
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)

    savedir='/home/data/maggie/mnist/mnist4stylegan2ada/datasets/viewmnist'
    os.makedirs(savedir,exist_ok=True)  
    device = torch.device('cuda')

    classification = ['zero','one','two','three','four','five','size','seven','eight','nine']
    for index, (imgs, labels) in enumerate(training_set):
        if index < 1000:
            print('index= %s' % index)
            print('one hot label= %s' % labels)    
            label = torch.argmax(torch.tensor(labels), -1)
            print('label= %s' % label)            

            img = torch.tensor(imgs,device=device)        
            img = img[-1]   
            Tesnsor_img = img.permute(1,0).clamp(0, 255).to(torch.uint8)  
            array_img = Tesnsor_img.cpu().numpy()                      
            PIL_image = PIL.Image.fromarray(array_img,'L')  

            target_fname = f'{savedir}/{index:08d}-{label:000d}-{classification[label]}.png'
            PIL_image.save(target_fname)
            print('target_fname=%s' % target_fname)
    
    print('Viewing *mnist* dataset finished !')
    return savedir

def ViewCIFAR10Train(target_dataset):
    data = target_dataset            
    assert data is not None
    assert isinstance(data, str)
    print('data=%s'%data)

    training_set_kwargs = utils.util.EasyDict(class_name='utils.training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False) 
    training_set = utils.util.construct_class_by_name(**training_set_kwargs) 
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)
    print('Label shape:', training_set.label_shape)

    savedir='/home/data/maggie/cifar10/cifar104stylegan2ada/datasets/viewcifar10'
    os.makedirs(savedir,exist_ok=True)  
    device = torch.device('cuda')

    classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for index, (imgs, labels) in enumerate(training_set):
        if index < 1000:
            print('index= %s' % index)
            print('one hot label= %s' % labels)    
            label = torch.argmax(torch.tensor(labels), -1)
            print('label= %s' % label)             

            img = torch.tensor(imgs,device=device)        
            print(img.shape)                                                
            Tesnsor_img = img.permute(1,2,0).clamp(0, 255).to(torch.uint8) 
            array_img = Tesnsor_img.cpu().numpy()                         
            PIL_image = PIL.Image.fromarray(array_img,'RGB')
            target_fname = f'{savedir}/{index:08d}-{label:000d}-{classification[label]}.png'
            PIL_image.save(target_fname)
            print('target_fname=%s' % target_fname)
    
    print('Viewing *cifar10* dataset finished !')
    return savedir
