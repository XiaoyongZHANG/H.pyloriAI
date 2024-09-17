################################################################
# 
##   論文用 様々なデータ拡張を単体で適応した場合の出力例を保存する ##
#
################################################################
import tensorflow as tf
import tensorflow.keras.preprocessing.image as image
import numpy as np 
import cv2, os

image_path = '/work/datasets/H.pylori_original/infected/5/0.jpg'
output_dir = '/work/output/pylori/misc/data_aug_example'
os.makedirs(output_dir, exist_ok=True)

img_original = image.load_img(path=image_path, target_size=(1024, 1024))
img_array = image.img_to_array(img_original)
img_expanded = np.expand_dims(img_array, axis=0)

generators = {
    'original': image.ImageDataGenerator(rescale = 1./255),
    'rotation': image.ImageDataGenerator(rescale = 1./255, rotation_range = 180, fill_mode = 'nearest'),
    'shift': image.ImageDataGenerator(rescale = 1./255, width_shift_range = 32, height_shift_range = 32, fill_mode = 'nearest'),
    'shear': image.ImageDataGenerator(rescale = 1./255, shear_range = 30, fill_mode = 'nearest'),
    'zoom': image.ImageDataGenerator(rescale = 1./255, zoom_range = [0.9, 1.1], fill_mode = 'nearest'),
    'flip': image.ImageDataGenerator(rescale = 1./255, horizontal_flip = True, vertical_flip = True, fill_mode = 'nearest'),
    'all': image.ImageDataGenerator(rescale = 1./255,
                rotation_range = 180,
                width_shift_range = 32,
                height_shift_range = 32,
                shear_range = 0.3,
                zoom_range = [0.9, 1.1],
                horizontal_flip = True,
                vertical_flip = True,
                fill_mode = 'nearest'
            )
}
    
for name in generators:

    output = generators[name].flow(x=img_expanded, batch_size=1)
    output_img = np.array(image.array_to_img(output[0][0]))
    cv2.imwrite(os.path.join(output_dir, name + '.jpg'), output_img)
 
