"""
Author: maggie
Date:   2021-06-15
Place:  Xidian University
@copyright
"""

import matplotlib.pyplot as plt

def SaveAccuracyCurve(model,dataset,exp_result_dir,global_train_acc,global_test_acc,png_name):
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc
    test_x = list(range(len(global_test_acc)))
    test_y = global_test_acc
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='green', label='on trainset accuracy') 
    plt.plot(test_x, test_y, color='blue', label='on testset accuracy') 
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()

def SaveLossCurve(model,dataset,exp_result_dir,global_train_loss,global_test_loss,png_name):
    train_x = list(range(len(global_train_loss)))
    train_y = global_train_loss
    test_x = list(range(len(global_test_loss)))
    test_y = global_test_loss
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='on trainset loss')                 
    plt.plot(test_x, test_y, color='red', label='on testset loss') 
 
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()