import torchvision
import torch

def check_file(file_path):
    if 'projected' in file_path:
        return True
    else:
        return False

def project_loader(file_path):
    with open file_path as f:
        raise NotImplementedError
        
dataset = torchvision.datasets.ImageFolder('/home/data/xieyi/project-kmnist-trainset-norm',is_valid_file=check_file,loader=project_loader)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,shuffle=True)

for x, y in dataloader:
    print(x,y)
    raise KeyError