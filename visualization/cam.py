################################################################
# 
##   以下の画像を生成するプログラム
#       1. オリジナル画像  :  original.jpg
#       2. ヒートマップ画像（32x32: 特徴マップの出力サイズ） :  heatmap.jpg
#       3. ヒートマップ画像（1024x1024: オリジナル画像に合わせてリサイズ） :  heatmap_resized.jpg
#       4. ヒートマップ画像（3.を色付けした画像） : heatmap_colorized.jpg
#       5. ヒートマップ画像（3.をオリジナル画像に重ねる） :  heatmap_overlayed.jpg
#       6. ヒートマップ画像（大津の二値化によって決定した閾値で二値化したヒートマップ画像） :  heatmap_binarized.jpg
#       7. オリジナル画像  （6.を用いてオリジナル画像をマスク処理） : original_masked_based_on_heatmap.jpg
#       8. アノテーション画像  （アノテーション（点群）から二値化画像を生成） : annotation_binarized.jpg
#       9. アノテーション画像  （8.をオリジナル画像に重ねる） : annotation_overlayed.jpg
#       10. ヒートマップ画像（8.を用いて3.からアノテーション領域のみを切り出す） :  heatmap_masked_based_on_annotation.jpg
#       11. ヒートマップ画像（8.を用いて3.からアノテーション領域以外を切り出す） :  heatmap_masked_based_on_not_annotation.jpg
#
################################################################

#########################################   オプション   ########################################################
model_base = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10'
dataset_base = "/work/datasets/H.pylori_k-fold/div4"
annotation_base = "/work/datasets/H.pylori_annotation/div4"
best_k = 3
img_size = 1024
################################################################################################################

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json, os, PIL, cv2, pathlib
from image_processor import ImageProcessor

best_model_path = os.path.join(model_base, 'training_result/' + str(best_k) + '/checkpoint/VGG16_GAP_best.h5')
dataset_path = os.path.join(dataset_base, str(best_k) + '/validation')
annotation_path = os.path.join(annotation_base, str(best_k))
dst = os.path.join(model_base, 'visualization')

os.makedirs(dst, exist_ok=True)

def get_contours(xml_path) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    contours = {}
    
    # object (ひだ、体部胃小区、前庭部胃小区)ごとにループ
    for obj in root.findall('object'):
        name: str = obj.find('name').text  # object名 (ひだ、体部胃小区、前庭部胃小区)を取得
        
        contours[name] = []  # 空の配列を用意
        
        # 各点の座標を取得
        for point in obj.find('polygon'):
            coord = point.text.split(',')  # x, y座標に分割
            contours[name].append(coord)

    # numpy 配列に変換
    for name in contours: contours[name] = np.array(contours[name], dtype=np.int32)
    return contours
    
def get_annotated_xml_img_path(base_path) -> dict:
    
    annotations = {'xml': {'infected': {}, 'non-infected': {}}, 'img': {'infected': {}, 'non-infected': {}}}
    
    for label in ["infected", "non-infected"]:
        
        for xml_file in pathlib.Path(os.path.join(base_path, "xml", label)).glob('*.xml'):
            annotations['xml'][label][xml_file.stem]: pathlib.PosixPath = xml_file  # xmlファイルの絶対パス  pathlibの型
            annotations['img'][label][xml_file.stem]: str = os.path.join(base_path, 'img', label, xml_file.stem + ".jpg")  # .jpgファイルの絶対パス
    return annotations

if __name__ == '__main__':
    
    image_processor = ImageProcessor()
    annotations = get_annotated_xml_img_path(annotation_path)
        
    # 畳み込み層までのモデルを作成
    model = tf.keras.models.load_model(best_model_path)
    model_conv = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer('block5_pool').output)
    
    # テストデータ 各症例の 0.jpg の場所を格納する
    img_paths: dict = image_processor.get_direction0_img_paths(base_path=dataset_path)
    
    # infected, non-infected それぞれで処理を行う
    for label in ['infected', 'non-infected']:
        
        # 症例ごとに繰り返す
        for case_num in tqdm(img_paths[label]):
            
            # 画像を読み込んで前処理
            img_original: PIL.Image.Image = tf.keras.preprocessing.image.load_img(path=img_paths[label][case_num], target_size=(img_size, img_size))
            img_array: np.ndarray = tf.keras.preprocessing.image.img_to_array(img_original)
            img_expanded = np.expand_dims(img_array, axis=0)
            img_normarized = img_expanded / 255.0 # 0-1で正規化
            
            # 各フィルターの出力
            activation = model_conv.predict(img_normarized) 
            
            # 空のヒートマップ 
            heatmap = np.zeros(dtype = np.float32, shape = (len(activation[0][:, 0, 0]), len(activation[0][:, 0, 0])))
            
            # 各フィルターについて計算
            for j in range(512):

                # 重みが正の時
                if model.get_layer("dense_3").get_weights()[0][j] > 0:
                    heatmap += activation[0][:, :, j] * model.get_layer("dense_3").get_weights()[0][j]
                else:
                    heatmap -= activation[0][:, :, j] * model.get_layer("dense_3").get_weights()[0][j]
        
            # ヒートマップを整える
            heatmap_normarized = image_processor.normarize_img(heatmap)  # 0-255で正規化
            heatmap_resized = cv2.resize(heatmap_normarized, (img_size, img_size))  # サイズを 元画像のサイズに
            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET) # カラーマップを適用    
            
            # 入力画像の用意
            img_array = img_array.astype(np.uint8)
            
            ALPHA=0.73
            heatmap_overlayed = cv2.addWeighted(img_array, ALPHA, heatmap_colored, 1 - ALPHA, 0)

            # 大津の二値化でヒートマップを二値化する
            heatmap_binarized = image_processor.binarize_heatmap(heatmap=heatmap_resized, cutoff='otsu')
            
            # 二値化したヒートマップ画像を使ってオリジナル画像をマスクする
            original_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            original_masked_based_on_heatmap = cv2.bitwise_and(original_gray, heatmap_binarized)
            
            # アノテーションした注目領域を読み込む
            contours = get_contours(xml_path=annotations['xml'][label][case_num])
            
            # アノテーションを描画する
            annotation_overlayed = image_processor.draw_annotations(img=img_array, contours=contours)
            
            annotation_binarized = image_processor.make_binarized_annotation_img(height=img_array.shape[0], width=img_array.shape[1], contours=contours)
            
            # 二値化したアノテーション画像を使ってヒートマップ画像をマスクする
            #heatmap_gray = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2GRAY)
            heatmap_gray = heatmap_resized
            heatmap_masked_based_on_annotation = cv2.bitwise_and(heatmap_gray, annotation_binarized)
            heatmap_masked_based_on_not_annotation = cv2.bitwise_and(heatmap_gray, cv2.bitwise_not(annotation_binarized))
            
            # 作成した画像の保存
            output_imgs: dict ={
                "original": img_array,
                "heatmap": heatmap, 
                "heatmap_resized": heatmap_resized, 
                "heatmap_colorized": heatmap_colored,
                "heatmap_overlayed": heatmap_overlayed,
                "heatmap_binarized": heatmap_binarized,
                "annotation_overlayed": annotation_overlayed,
                "annotation_binarized": annotation_binarized,
                "original_masked_based_on_heatmap": original_masked_based_on_heatmap,
                "heatmap_masked_based_on_annotation": heatmap_masked_based_on_annotation,
                "heatmap_masked_based_on_not_annotation": heatmap_masked_based_on_not_annotation
            }
            
            # 予測
            predict = model.predict(img_normarized)
    
            
            # 画像保存用
            cam_output_dir = os.path.join(dst, label, '{:.15f}'.format(predict[0][0]) + '_' + str(case_num))
            os.makedirs(cam_output_dir, exist_ok=True)
            
            image_processor.save_imgs_from_ndarray(imgs_dict=output_imgs, output_dir=cam_output_dir)
