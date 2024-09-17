import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import numpy as np
import japanize_matplotlib  # matplotlib の日本語出力ライブラリ 


class Plotter():

    def plot_acc_loss(self, history, file_name):

        acc = history['acc']
        val_acc = history['val_acc']
        loss = history['loss']
        val_loss = history['val_loss']
        epochs = range(1, len(acc) +1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True, squeeze=False) # squeeze : axesを常に2次元配列として扱う

        # acc のプロット
        ax[0, 0].set_xlabel("epoch", size=14)
        ax[0, 0].set_ylabel("正\n解\n率", size=14, labelpad=15, rotation=0, va="center")
        ax[0, 0].minorticks_on()  # 補助目盛を表示
        ax[0, 0].grid() # 目盛り線を表示
        ax[0, 0].set_ylim(0.48, 1.0)

        ax[0, 0].plot(epochs, acc, "-o", label="training acc", markersize=7)  # - : 実線,  o : 丸,  ^ : 三角
        ax[0, 0].plot(epochs, val_acc, "-^", label="validation acc", markersize=7)

        ax[0, 0].legend() # 凡例を表示

        # loss のプロット
        ax[0, 1].set_xlabel("epoch", size=14)
        ax[0, 1].set_ylabel("損\n失", size=14, labelpad=15, rotation=0, va="center")
        ax[0, 1].minorticks_on()  # 補助目盛を表示
        ax[0, 1].grid() # 目盛り線を表示
        ax[0, 1].set_ylim(0, 1.4)           

        ax[0, 1].plot(epochs, loss, "-o", label="training loss", markersize=7)  # - : 実線,  o : 丸,  ^ : 三角
        ax[0, 1].plot(epochs, val_loss, "-^", label="validation loss", markersize=7)

        ax[0, 1].legend() # 凡例を表示

        fig.savefig(file_name)

    def plot_roc_cross_val(self, dataset_info, aug_params, train_params, test_dirs, file_name, model_paths):

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True, squeeze=False) # squeeze : axesを常に2次元配列として扱う

        i = 0
        fprs = []
        tprs = []
        aucs = []

        # 作成したモデル分繰り返す
        for (model_path, test_dir) in zip(model_paths, test_dirs):

            model = tf.keras.models.load_model(model_path)
            test_datagen = ImageDataGenerator(rescale = aug_params["rescale"])
            test_generator = test_datagen.flow_from_directory(
                test_dir, 
                target_size = dataset_info["image_size"],
                batch_size=1,
                class_mode='binary',
                shuffle=False
            )
            pred = model.predict(test_generator).ravel()
            fpr, tpr, thresholds = roc_curve(test_generator.classes, pred)
            fprs.extend(fpr)
            tprs.extend(tpr)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)

            # ROC のプロット
            ax[0, 0].set_xlabel("偽陽性率[%]", size=14)
            ax[0, 0].set_ylabel("真\n陽\n性\n率\n[%]", size=14, labelpad=15, rotation=0, va="center")

            ax[0, 0].plot(fpr, tpr, "-^", label="ROC fold"+str(i)+": AUC=%.4f "%auc, markersize=2, alpha=0.2)  # - : 実線,  o : 丸,  ^ : 三角

            i += 1

        fprs.sort()
        tprs.sort()
        auc = metrics.auc(fprs, tprs)

        # ROC のプロット
        ax[0, 0].plot(fprs[::4], tprs[::4], "-o", label="AUC 平均　　: %.4f "%auc + "\nAUC 標準偏差: %.4f "%np.std(aucs), markersize=3, linewidth=1.0, color='red')  # - : 実線,  o : 丸,  ^ : 三角
        #ax[0, 0].plot(fprs[::4], tprs[::4], "-o", label="ROC Avg:         AUC = %.4f "%auc, markersize=5, linewidth=2.0, color='red')  # - : 実線,  o : 丸,  ^ : 三角
        ax[0, 0].set_xticks(np.arange(0, 1.1, step=0.1))
        ax[0, 0].set_yticks(np.arange(0, 1.1, step=0.1))
        ax[0, 0].set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax[0, 0].set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        # 分散のプロット
        #ax[0, 0].plot([], [], marker="s", color='red', label="AUC 標準偏差: %.6f "%np.std(aucs), linestyle="none", markersize=1)
        #ax[0, 0].plot([], [], marker="s", color='red', label="AUC Std:                 %.6f "%np.std(aucs), linestyle="none", markersize=1)

        #ax[0, 0].minorticks_on()  # 補助目盛を表示
        ax[0, 0].grid()           # 目盛り線を表示
        ax[0, 0].legend(fontsize=12)         # 凡例を表示

        fig.savefig(file_name, dpi=300)


    def plot_roc(self, dataset_info, aug_params, train_params, file_name, model):

        test_datagen = ImageDataGenerator(rescale = aug_params["rescale"])

        test_generator = test_datagen.flow_from_directory(
            train_params["valid_dir"], 
            target_size = dataset_info["image_size"],
            batch_size=1,
            class_mode='binary',
            shuffle=False
        )

        pred = model.predict(test_generator).ravel()
        fpr, tpr, thresholds = roc_curve(test_generator.classes, pred)
        auc = metrics.auc(fpr, tpr)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), tight_layout=True, squeeze=False) # squeeze : axesを常に2次元配列として扱う

        # ROC のプロット
        ax[0, 0].set_xlabel("偽陽性率[%]", size=14)
        ax[0, 0].set_ylabel("真\n陽\n性\n率\n[%]", size=14, labelpad=15, rotation=0, va="center")
        
        # プロット
        ax[0, 0].plot(fpr, tpr, "-o", label="ROC Curve: AUC = %.4f"%auc, markersize=2)  # - : 実線,  o : 丸,  ^ : 三角

        ax[0, 0].set_xticks(np.arange(0, 1.1, step=0.1))
        ax[0, 0].set_yticks(np.arange(0, 1.1, step=0.1))
        ax[0, 0].set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax[0, 0].set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        #ax[0, 0].minorticks_on()  # 補助目盛を表示
        ax[0, 0].grid()           # 目盛り線を表示
        ax[0, 0].legend(fontsize=12)         # 凡例を表示

        fig.savefig(file_name, dpi=300)