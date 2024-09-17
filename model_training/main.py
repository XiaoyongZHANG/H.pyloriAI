from model import VGG16GAPModel, VGG16Model
from plotter import Plotter

import os, pickle, datetime

#########################################   オプション   ########################################################
DATASET_INFO = {
    "dataset_base_path": "/work/datasets/H.pylori_k-fold/div4",  # 自分の環境に合わせてパスを変更する
    "total_images": 1600,        # 画像枚数
    "div_num": 4,                # 交差検証の k=4 を指定
    "image_size": (1024, 1024)   # 学習時の画像サイズ. 先行研究では (320, 320). 本研究では (1024, 1024)
}   

# データ拡張のパラメータを指定
# 詳細は https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# もしくは 専攻論文を参照
AUG_PARAMS = {
    "rescale": 1./255,      
    "rotation_range": 180,
    "width_shift_range": 32,
    "height_shift_range": 32,
    "shear_range": 0.3,
    "zoom_range": [0.9, 1.1],
    "horizontal_flip": True,
    "vertical_flip": True,
    "fill_mode": 'nearest',
}

TRAIN_PARAMS = {
    "name": "VGG16_GAP",    # モデルを指定 (VGG16, VGG16_GAP) のどちらか
    "train_dir": "",        # 交差検証では指定しない
    "valid_dir": "",        # 交差検証では指定しない
    "batch_size": 16,       # GPU のメモリサイズに合わせて調整するf
    "epochs": 3           # 200, 400 あたりがおすすめ. val_loss がまだ小さくなるようであれば更に回数を増やす.
}

# 学習結果の保存先
now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
OUTPUT_BASE = f'/work/output/pylori/{TRAIN_PARAMS["name"]}/div{str(DATASET_INFO["div_num"])}/{str(DATASET_INFO["image_size"][0])}x{str(DATASET_INFO["image_size"][1])}_batchsize{str(TRAIN_PARAMS["batch_size"])}_epochs{str(TRAIN_PARAMS["epochs"])}/{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}'
##############################################################################################################

training_result_path = os.path.join(OUTPUT_BASE, 'training_result')

if __name__ == "__main__":

    # k=4 の場合 dirs = [0, 1, 2, 3]
    dirs = os.listdir(DATASET_INFO["dataset_base_path"])
    
    for dir in dirs:
   
        # インスタンス化
        if TRAIN_PARAMS["name"] == "VGG16_GAP": model = VGG16GAPModel()
        elif TRAIN_PARAMS["name"] == "VGG16": model = VGG16Model()
        
        plotter = Plotter()

        # block5_conv1 以前を凍結、以降を訓練可能なモデルを作成（転移学習）
        model.create_model(input_shape = DATASET_INFO["image_size"], trainable_base_layer = 'block5_conv1')      

        # 訓練用、検証用データのパスをセット
        TRAIN_PARAMS["train_dir"] = os.path.join(DATASET_INFO["dataset_base_path"], dir, 'train')
        TRAIN_PARAMS["valid_dir"] = os.path.join(DATASET_INFO["dataset_base_path"], dir, 'validation')

        # チェックポイントの保存先を作成
        output_dir = os.path.join(training_result_path, dir)
        checkpoint_dir = os.path.join(output_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # モデル 訓練
        model.fit(DATASET_INFO, AUG_PARAMS, TRAIN_PARAMS, checkpoint_dir) 
        
        # モデル 保存
        model_dir = os.path.join(output_dir, TRAIN_PARAMS["name"]+".h5")
        model.built_model.save(model_dir)

        # history 保存
        history_dir = os.path.join(output_dir, "History")
        with open(history_dir, 'wb') as file:
            pickle.dump(model.history.history, file)

        # ACC LOSS 出力
        plotter.plot_acc_loss(
            history=model.history.history, 
            file_name=os.path.join(output_dir, TRAIN_PARAMS["name"]+"_acc_loss.jpg")
        )

        # ROC 出力
        plotter.plot_roc(
            DATASET_INFO, AUG_PARAMS, TRAIN_PARAMS,
            file_name=os.path.join(output_dir, TRAIN_PARAMS["name"]+"_roc.jpg"), 
            model=model.built_model
        )


    # 平均の ROC AUCを出力
    plotter = Plotter()
    model_paths = []
    test_dirs = []
    for dir in dirs: 
        model_paths.append(os.path.join(training_result_path, dir, "checkpoint", TRAIN_PARAMS["name"]+"_best.h5"))
        test_dirs.append(os.path.join(DATASET_INFO["dataset_base_path"], dir, "validation"))

    plotter.plot_roc_cross_val(
        DATASET_INFO, AUG_PARAMS, TRAIN_PARAMS, 
        test_dirs = test_dirs,
        file_name=os.path.join(training_result_path, TRAIN_PARAMS["name"]+"_roc_cross_val.jpg"), 
        model_paths=model_paths
    )
    

    print("successfully finished")  