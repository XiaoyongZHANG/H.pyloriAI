import os, cv2
import numpy as np

class ImageProcessor():
    def __init__(self):
        pass
        
    def get_direction0_img_paths(self, base_path: str) -> dict:
        
        img_paths = {'infected':{}, 'non-infected':{}}
        for label in img_paths:

            case_list: list = os.listdir(os.path.join(base_path, label))             
            for case in case_list: img_paths[label][case] = os.path.join(base_path, label, case, '0.jpg')
            
        return img_paths 
    
    def normarize_img(self, img: np.ndarray):
        img = img - np.min(img)             # 最小値が 0 になるように移動
        if np.max(img) != 0:
            img = img / np.max(img)             # 0-1 の間に正規化
        img = (img * 255).astype("uint8")   # 0-255 に  float から int に
        return img
    
    
    def save_imgs_from_ndarray(self, imgs_dict: dict, output_dir: str):
        
        # ヒートマップの種類ごとにループを回して各フォルダへ保存
        for name in imgs_dict:
            
            # ヒートマップの種類ごとに保存する
            output_path = os.path.join(output_dir, name + ".jpg")
            cv2.imwrite(output_path, imgs_dict[name])
            
            
    def binarize_heatmap(self, heatmap, cutoff='otsu'):
        
        #gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)                # グレースケール変換
        gray = heatmap
        
        if cutoff == 'otsu':
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 閾値で2値化
        else:
            ret, thresh = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)  # 閾値で2値化
            
        return thresh
    
    def draw_annotations(self, img, contours, thickness=-1, alpha=0.73) -> np.ndarray:
        
        # 描画用
        height, width, channel = img.shape
        img_contours = np.zeros((height, width, channel), dtype=np.uint8)
        
        for name in contours:
        
            # 輪郭を描画（塗りつぶし）
            cv2.drawContours(image=img_contours, contours=[contours[name]], contourIdx=0, color=(255, 0, 0), thickness=thickness)
            
            # 輪郭の点を描画
            for point in contours[name]:
                cv2.circle(img=img_contours, center=tuple(point), radius=5, color=(255, 150, 0), thickness=-1)
            
            # 領域の名前を描画
            # 重心を計算
            m = cv2.moments(contours[name])
            x, y= m['m10']/m['m00'] , m['m01']/m['m00']
            cv2.putText(img_contours, text=name, org=(int(x)-70, int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.3, color=(255, 255, 0), thickness=5)
            
        return cv2.addWeighted(img, alpha, img_contours, 1 - alpha, 0)
              
              
    def make_binarized_annotation_img(self, height, width, contours):
        
        # cv2.drawContoursがRGB画像のみ対応のため3チャンネルで作成
        img_contours = np.zeros((height, width, 3), dtype=np.uint8)
        
        for name in contours:
        
            # 輪郭を描画（塗りつぶし）
            cv2.drawContours(image=img_contours, contours=[contours[name]], contourIdx=0, color=(255, 255, 255), thickness=-1)
        
        img_gray = cv2.cvtColor(img_contours, cv2.COLOR_BGR2GRAY)
        
        return img_gray