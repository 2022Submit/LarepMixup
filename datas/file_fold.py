import os
import shutil
from tqdm import trange

original_path = "/home/data/xieyi/project-kmnist-trainset"
new_path = "/home/data/xieyi/project-kmnist-trainset-norm"

def norm_folder(files_path):
    file_list = next(os.walk(files_path))[2]
    n_file = len(file_list)
    for i in trange(n_file):
        name = file_list[i]
        id, class_id, class_name, type_name = name.split('-')
        type_name = type_name.split('.')[0]
        if type_name != 'projected_w':
            continue
        file_path = os.path.join(original_path,name)
        target_path = os.path.join(new_path,class_name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        shutil.copy(file_path,target_path)

norm_folder(original_path)