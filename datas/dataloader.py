

import torch.utils.data
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
import robustness.datasets


class RMDataloader:
    def __init__(self,args,traindataset,testdataset) -> None:
        self._args = args    
        self._traindataset = traindataset
        self._testdataset = testdataset        
        self._traindataloader = self.__loadtraindataloader__()
        self._testdataloader = self.__loadtestdataloader__()

    def traindataloader(self)->"torch.utils.data.DataLoader":
        return self._traindataloader
    
    def testdataloader(self)->"torch.utils.data.DataLoader":
        return self._testdataloader

    def __loadtraindataloader__(self) ->"torch.utils.data.DataLoader":
        if self._args.dataset != 'imagenetmixed10':
            train_dataloader = torch.utils.data.DataLoader(   
                self._traindataset,
                batch_size=self._args.batch_size,
                shuffle=True,                                  
                num_workers=self._args.cpus,
                pin_memory=True,
            )
        elif self._args.dataset == 'imagenetmixed10':
            batch_size=self._args.batch_size
            num_workers=self._args.cpus
            train_loader, test_loader = self._traindataset.make_loaders(workers=num_workers, batch_size=batch_size, shuffle_train=True)  
            train_dataloader = train_loader

        print(f'Loading *{self._args.dataset}* train dataloader finished !')
        return train_dataloader
    
    def __loadtestdataloader__(self):
        if self._args.dataset != 'imagenetmixed10':
            test_dataloader = torch.utils.data.DataLoader(                       
                self._testdataset,
                batch_size=self._args.batch_size,                                  
                shuffle=False,
                num_workers=self._args.cpus,
                pin_memory=True,
            )
        elif self._args.dataset == 'imagenetmixed10':
            batch_size=self._args.batch_size
            num_workers=self._args.cpus
            train_loader, test_loader = self._traindataset.make_loaders(workers=num_workers, batch_size=batch_size, shuffle_val=False)
            test_dataloader = test_loader

        print(f'Loading *{self._args.dataset}* test dataloader finished !')
        return test_dataloader        
