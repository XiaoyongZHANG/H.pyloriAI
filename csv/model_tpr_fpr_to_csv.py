#######################################################################
# 
##   .h5 ファイルからモデルの各閾値におけるtpr, fprを計算し.csvに出力する ##
#
#######################################################################

#########################################   オプション   ########################################################
model_base = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10'
model_name = 'VGG16_GAP'
dataset_base = "/work/datasets/H.pylori_k-fold/div4"
img_size=1024
################################################################################################################

import os
import pickle
import csv
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
import numpy as np
import tqdm

histories = {"0": os.path.join(model_base, 'training_result/0/checkpoint/' + model_name + '_best.h5'),
             "1": os.path.join(model_base, 'training_result/1/checkpoint/' + model_name + '_best.h5'), 
             "2": os.path.join(model_base, 'training_result/2/checkpoint/' + model_name + '_best.h5'), 
             "3": os.path.join(model_base, 'training_result/3/checkpoint/' + model_name + '_best.h5')}


test_dirs = {"0": os.path.join(dataset_base, '0', 'validation'),
             "1": os.path.join(dataset_base, '1', 'validation'), 
             "2": os.path.join(dataset_base, '2', 'validation'), 
             "3": os.path.join(dataset_base, '3', 'validation')}

dst_tpr_fpr = os.path.join(model_base, 'csv/tpr_fpr')
dst_pred = os.path.join(model_base, 'csv/prediction')
os.makedirs(dst_tpr_fpr, exist_ok=True)
os.makedirs(dst_pred, exist_ok=True)

# 作成したモデル分繰り返す
for k in tqdm.tqdm(histories):

    model = tf.keras.models.load_model(histories[k])
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dirs[k], 
        target_size = (img_size, img_size),
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )
    pred = model.predict(test_generator).ravel()
    fpr, tpr, threshold = roc_curve(test_generator.classes, pred)
    youden = tpr-fpr

    #fpr.sort()
    #tpr.sort()
    
    fpr = fpr.tolist()
    tpr = tpr.tolist()
    youden = youden.tolist()
    fpr = ["fpr"] + fpr
    tpr = ["tpr"] + tpr
    youden = ['tpr-fpr'] + youden
    threshold = threshold.tolist()
    threshold = ["threshold"] + threshold
    
    ground_truth = test_generator.classes.tolist()
    ground_truth = ['ground_truth'] + ground_truth
    pred = pred.tolist()
    pred = ['pred'] + pred
        
    # Ground Truth と predの値の保存

    with open(os.path.join(dst_pred, k + '_prediction.csv'), mode='w') as csv_file:
            
        for data in [ground_truth, pred]:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(data)       
    
    # fpr, tpr の保存
    with open(os.path.join(dst_tpr_fpr, k + '_tpr_fpr.csv'), mode='w') as csv_file:
            
        for data in [fpr, tpr, threshold, youden]:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(data)            
