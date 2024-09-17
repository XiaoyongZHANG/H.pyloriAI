####################################################################
# 
##   cam.py で出力された画像をもとに Drop%, GI, BIを計算し.csvに出力 ##
#
####################################################################

#########################################   オプション   ########################################################
model_base = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10'
best_k = 3
################################################################################################################

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json, os, PIL, cv2, pathlib, csv, statistics

cam_result = os.path.join(model_base, 'visualization')
csv_dst = os.path.join(model_base, 'csv/drop_gi_bi')
model_path = os.path.join(model_base, 'training_result/' + str(best_k) + '/checkpoint/VGG16_GAP_best.h5')
    
# モデルの読み込み
model = tf.keras.models.load_model(model_path)
    
if __name__ == '__main__':
    
    drop_all = []
    gi_all = []
    bi_all = []
    
    # infected, non-infected それぞれで処理を行う
    for label in ['infected', 'non-infected']:
        
        case_nums = os.listdir(os.path.join(cam_result, label))
        
        drop_gi_bi = {'Pred_original': {'name': 'Pred_original'}, 'Pred_masked': {'name': 'Pred_masked'}, 'Drop': {'name': 'Drop'}, 'Increase':{'name': 'Increase'}, 'GI': {'name': 'GI'}, 'BI': {'name': 'BI'}}
        
        # 症例ごとに繰り返す
        for case_num in tqdm(case_nums):

            img_original = cv2.imread(os.path.join(cam_result, label, case_num, 'original.jpg'))
            img_heatmap = cv2.imread(os.path.join(cam_result, label, case_num, 'heatmap_resized.jpg'))
            original_masked = cv2.imread(os.path.join(cam_result, label, case_num, 'original_masked_based_on_heatmap.jpg'))
            heatmap_masked_gi = cv2.imread(os.path.join(cam_result, label, case_num, 'heatmap_masked_based_on_annotation.jpg'))
            heatmap_masked_bi = cv2.imread(os.path.join(cam_result, label, case_num, 'heatmap_masked_based_on_not_annotation.jpg'))
            
            ################### Drop の計算 ####################
            
            original: np.ndarray = tf.keras.preprocessing.image.img_to_array(img_original)
            original = np.expand_dims(original, axis=0)
            original /= 255.0 # 正規化
            masked: np.ndarray = tf.keras.preprocessing.image.img_to_array(original_masked)
            masked = np.expand_dims(masked, axis=0)
            masked /= 255.0 # 正規化
            
            # 予測
            pred_original = model.predict(original)[0][0]
            pred_masked = model.predict(masked)[0][0]
            
            # ラベルが陽性の場合 0-1を入れ替える
            if label == 'infected':
                pred_original = 1 - pred_original
                pred_masked = 1 - pred_masked
    
            # Drop の計算
            drop = (max(0, pred_original - pred_masked) / pred_original ) * 100
            drop_gi_bi['Pred_original'][case_num] = pred_original
            drop_gi_bi['Pred_masked'][case_num] = pred_masked
            drop_gi_bi['Drop'][case_num] = drop
            
            
            ##################### GTC_intensity, BC_intensity の計算 #########################
            drop_gi_bi['GI'][case_num] = (np.sum(heatmap_masked_gi) / 255 ) / np.count_nonzero(heatmap_masked_gi)
            drop_gi_bi['BI'][case_num] = (np.sum(heatmap_masked_bi) / 255) / np.count_nonzero(heatmap_masked_bi)

            
        drop_list = list(drop_gi_bi['Drop'].values())[1:]
        gi_list = list(drop_gi_bi['GI'].values())[1:]
        bi_list = list(drop_gi_bi['BI'].values())[1:]
        
        drop_all.extend(drop_list)
        gi_all.extend(gi_list)
        bi_all.extend(bi_list)
        
        print('\n' +label + '\nDrop: ' + str(statistics.mean(drop_list)) + ' ± ' + str(statistics.pstdev(drop_list)) + 
              '\nGI: ' + str(statistics.mean(gi_list)) + ' ± ' + str(statistics.pstdev(gi_list)) +
              '\nBI: ' + str(statistics.mean(bi_list)) + ' ± ' + str(statistics.pstdev(bi_list)) )

        os.makedirs(csv_dst, exist_ok=True)
        data = [drop_gi_bi['Pred_original'], drop_gi_bi['Pred_masked'], drop_gi_bi['Drop'], drop_gi_bi['GI'], drop_gi_bi['BI']]
        with open(os.path.join(csv_dst, label + '_Drop_GI_BI.csv'), mode='w') as csv_file:
            field_name = []
            for key in drop_gi_bi['Drop'].keys():
                field_name.append(key)

            writer = csv.DictWriter(csv_file, fieldnames=field_name)
            writer.writeheader()
            writer.writerows(data)  

        
    print('\nAll' + '\nDrop%: ' + str(statistics.mean(drop_all)) + ' ± ' + str(statistics.pstdev(drop_all)) + 
        '\nGI: ' + str(statistics.mean(gi_all)) + ' ± ' + str(statistics.pstdev(gi_all)) +
        '\nBI: ' + str(statistics.mean(bi_all)) + ' ± ' + str(statistics.pstdev(bi_all)) )
            
   