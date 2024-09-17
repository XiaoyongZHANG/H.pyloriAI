#####################################################################
# 
##   学習時の履歴 History から各EpochにおけるACC, LOSSを.csvに出力する ##
#
#####################################################################

#########################################   オプション   ########################################################
base = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10'
################################################################################################################

import os
import pickle
import csv

src = os.path.join(base, 'training_result')     # 変換前パス
dst = os.path.join(base, 'csv/acc_loss')        # 変換後パス
os.makedirs(dst, exist_ok=True)

histories = {"0": os.path.join(src, '0', 'History'),
             "1": os.path.join(src, '1', 'History'), 
             "2": os.path.join(src, '2', 'History'), 
             "3": os.path.join(src, '3', 'History')}

for k in histories:

    # acc と loss の history を load
    with open(histories[k], 'rb') as history_file:
        history=pickle.load(history_file)  

        epochs = ['epochs'] + list(range(1, len(history['acc'])+1))
        acc = ['acc'] + history['acc']
        val_acc = ['val_acc'] + history['val_acc']
        loss = ['loss'] + history['loss']
        val_loss = ['val_loss'] + history['val_loss']
        
        # 書き込み先 open
        with open(os.path.join(dst, k + '_acc_loss.csv'), mode='w') as csv_file:
            
            for data in [epochs, acc, val_acc, loss, val_loss]:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(data)            
            