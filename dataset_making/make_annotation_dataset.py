#################################################################################
# 
##   最も性能が良かったkを指定し、アノテーション用に0.jpgのみのデータセットを作成する ##
#
#################################################################################

BASE = '/work/datasets/H.pylori K-Fold/div4/3/validation'
OUTPUT = '/work/datasets/H.pylori annotation/div4/3'

import os, shutil

for mode in ['infected', 'non-infected']:
    
    base_path = os.path.join(BASE, mode)
    os.makedirs(os.path.join(OUTPUT, mode), exist_ok=True)

    dir_nums: list = os.listdir(base_path)
    
    for dir_num in dir_nums: 
        file_path = (os.path.join(base_path, dir_num, "0.jpg"))

        shutil.copyfile(file_path, os.path.join(OUTPUT, mode, dir_num + ".jpg"))