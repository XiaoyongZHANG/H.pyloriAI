from tensorflow.keras.applications import VGG16 
from keras import layers
from keras import models
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers 
from keras.preprocessing.image import ImageDataGenerator

import os

# 継承元クラス
class Model:

    def __init__(self):
        self.built_model = None
        self.history = None


    # レイヤーの凍結を行う関数
    def set_layers_trainable(self, model, trainable_base_layer):
        set_trainable = False
        for layer in model.layers:

            if layer.name == trainable_base_layer:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else: 
                layer.trainable = False

        return model


    # モデルの学習を行う関数
    def fit(self, dataset_info, aug_params, train_params, checkpoint_dir):

        # 訓練データのジェネレーター生成
        train_datagen = ImageDataGenerator(
            rescale = aug_params["rescale"],
            rotation_range = aug_params["rotation_range"],
            width_shift_range = aug_params["width_shift_range"],
            height_shift_range = aug_params["height_shift_range"],
            shear_range = aug_params["shear_range"],
            zoom_range = aug_params["zoom_range"],
            horizontal_flip = aug_params["horizontal_flip"],
            vertical_flip = aug_params["vertical_flip"],
            fill_mode = aug_params["fill_mode"],
        )

        # 検証データのジェネレーター生成（訓練データはデータ拡張しない）
        validation_datagen = ImageDataGenerator(rescale = aug_params["rescale"])

        train_generator = train_datagen.flow_from_directory(
            train_params["train_dir"],
            target_size = dataset_info["image_size"],
            batch_size = train_params["batch_size"],
            class_mode = 'binary',
            seed = 0
            #color_mode="grayscale"
        )
            
        validation_generator = validation_datagen.flow_from_directory(
            train_params["valid_dir"],
            target_size = dataset_info["image_size"],
            batch_size = train_params["batch_size"],
            class_mode = 'binary'
            #color_mode="grayscale"
        )

        # 最も性能の良いモデルを保存するために保存条件を設定する
        #### 参考 : https://qiita.com/tom_eng_ltd/items/7ae0814c2d133431c84a
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, train_params["name"]+"_best.h5"), 
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            period=1
        )
        
        # 学習
        test_num = dataset_info["total_images"] / dataset_info["div_num"]      # 検証データの枚数
        self.history = self.built_model.fit_generator(
            train_generator,
            steps_per_epoch = int((dataset_info["total_images"]-test_num)/train_params["batch_size"]),
            epochs=train_params["epochs"],
            validation_data=validation_generator,
            validation_steps = int((test_num)/train_params["batch_size"]),
            callbacks=[checkpoint]
        )

    def create_model(self, input_shape, trainable_base_layer):

        conv = VGG16(
            weights = 'imagenet',      # 転移学習で使用する事前学習済みの重み
            include_top = False,
            input_shape = (input_shape[0], input_shape[1], 3)
        )
        
        # 空のモデルを作成
        model = models.Sequential() 

        # レイヤーを追加
        model = self.add_layers(model, conv)

        # レイヤーの凍結、解凍をセットする
        model = self.set_layers_trainable(model, trainable_base_layer)
        
        # モデルをコンパイル
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
            metrics=['acc'])

        self.built_model = model


class VGG16Model(Model): 

    #def __init__(self, input_shape):
      
    def add_layers(self, model, conv):

        # レイヤーをバラして追加（一層ごとに凍結するかどうかを指定できるため）
        for layer in conv.layers:
            model.add(layer)
        
        # 全結合層を追加
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
 
        return model
           



class VGG16GAPModel(Model): 

    #def __init__(self):
    
    def add_layers(self, model, conv):

        # レイヤーをバラして追加
        for layer in conv.layers:
            model.add(layer)
            
        # 畳み込み層を追加（試験的に畳み込み層を増やしてみたが、効果なし）
        """
        model.add(layers.Conv2D(1024, (3, 3), padding="same", activation='relu'))
        model.add(layers.Conv2D(1024, (3, 3), padding="same", activation='relu'))
        model.add(layers.Conv2D(1024, (3, 3), padding="same", activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        """
        # GAP層 を追加
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(1, activation='sigmoid'))
 
        return model


