import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json, os, PIL, cv2, pathlib, csv

from image_processor import ImageProcessor

#########################################   オプション   ########################################################
model_base = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10'
annotation_base = "/work/datasets/H.pylori_annotation/div4/3"
################################################################################################################

cam_result = os.path.join(model_base, 'visualization')
csv_dst = os.path.join(model_base, 'csv')


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
    annotations = get_annotated_xml_img_path(annotation_base)
    
    # infected, non-infected それぞれで処理を行う
    for label in ['infected', 'non-infected']:
        
        case_nums = os.listdir(os.path.join(cam_result, label))
        
        shared_interest = {'IoU': {'name': 'IoU'} , 'GTC': {'name': 'GTC'}, 'SC': {'name': 'SC'}}
        
        # 症例ごとに繰り返す
        for case_num in tqdm(case_nums):
            
            original_path = os.path.join(cam_result, label, case_num, 'original.jpg')
            img_original = cv2.imread(original_path)
            heatmap_path = os.path.join(cam_result, label, case_num, 'heatmap_resized.jpg')
            img_heatmap = cv2.imread(heatmap_path)
            
            # 大津の二値化でヒートマップを二値化する
            heatmap_binarized = image_processor.binarize_heatmap(heatmap=img_heatmap, cutoff='otsu')
            
            # 二値化したヒートマップ画像を使ってオリジナル画像をマスクする
            original_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            original_masked_based_on_heatmap = cv2.bitwise_and(original_gray, heatmap_binarized)
            
            # アノテーションした注目領域を読み込む
            contours = get_contours(xml_path=annotations['xml'][label][case_num.split('_')[1]])
            
            # アノテーションを描画する
            annotation_overlayed = image_processor.draw_annotations(img=img_original, contours=contours)
            
            annotation_binarized = image_processor.make_binarized_annotation_img(height=img_original.shape[0], width=img_original.shape[1], contours=contours)
            
            # 二値化したアノテーション画像を使ってヒートマップ画像をマスクする
            heatmap_gray = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2GRAY)
            heatmap_masked_based_on_annotation = cv2.bitwise_and(heatmap_gray, annotation_binarized)
            heatmap_masked_based_on_not_annotation = cv2.bitwise_and(heatmap_gray, cv2.bitwise_not(annotation_binarized))
            
            # 作成した画像の保存
            output_imgs: dict ={
                "heatmap_binarized": heatmap_binarized,
                "annotation_overlayed": annotation_overlayed,
                "annotation_binarized": annotation_binarized,
                "original_masked_based_on_heatmap": original_masked_based_on_heatmap,
                "heatmap_masked_based_on_annotation": heatmap_masked_based_on_annotation,
                "heatmap_masked_based_on_not_annotation": heatmap_masked_based_on_not_annotation
            }
            
            image_processor.save_imgs_from_ndarray(imgs_dict=output_imgs, output_dir=os.path.join(cam_result, label, case_num))
