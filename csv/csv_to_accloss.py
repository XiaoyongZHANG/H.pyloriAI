################################################################
# 
##         .csvファイルから ACC, LOSS のグラフを出力する        ##
#
################################################################

#########################################   オプション   ########################################################
base = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10'
name = '3_acc_loss'
lang_jp = True     # 日本語で書き込み
################################################################################################################

import matplotlib.pyplot as plt
import os
import pandas as pd
import japanize_matplotlib

src = os.path.join(base, 'csv/acc_loss/' + name + '.csv')
dst = os.path.join(base, 'graph/acc_loss/' + name + '.jpg')
os.makedirs(os.path.join(base, 'graph/acc_loss'), exist_ok=True)

# プロット用
fig, ax = plt.subplots(1, 2, figsize=(12, 5), tight_layout=True, squeeze=False) # squeeze : axesを常に2次元配列として扱う
ax[0, 0].set_xlabel('epoch', size=14)
ax[0, 0].minorticks_on()                # 補助目盛を表示
ax[0, 0].grid()                         # 目盛り線を表示

ax[0, 1].set_xlabel('epoch', size=14)
ax[0, 1].minorticks_on()                # 補助目盛を表示
ax[0, 1].grid()                         # 目盛り線を表示       

if lang_jp:
    ax[0, 0].set_ylabel('正\n解\n率', size=14, labelpad=15, rotation=0, va='center')
    ax[0, 1].set_ylabel('損\n失', size=14, labelpad=15, rotation=0, va='center')
else:
    ax[0, 0].set_ylabel('accuracy', size=14)
    ax[0, 1].set_ylabel('loss', size=14)
            
# CSVファイルの読み込み
data = pd.read_csv(src, header=None).T

# データの整形
data.columns = data.iloc[0]
data = data[1:]
data.index = pd.to_numeric(data.index)
data = data.apply(pd.to_numeric)
data = data.reset_index().rename(columns={'index': 'epoch'})

acc = data['acc'].values
val_acc = data['val_acc'].values
loss = data['loss'].values
val_loss = data['val_loss'].values
epochs = data['epoch'].values        
        
# プロット
if lang_jp:
    label_train_acc = '本研究モデル 1024×1024 学習　 正解率'
    label_train_loss = '本研究モデル 1024×1024 学習　 損失'
    label_valid_acc = '本研究モデル 1024×1024 テスト 正解率'
    label_valid_loss = '本研究モデル 1024×1024 テスト 損失'
else:
    # 要修正
    label_train_acc = 'Training accuracy'
    label_train_loss = 'Training loss'
    label_valid_acc = 'Validation accuracy'
    label_valid_loss = 'Validation loss'

# 本研究 1024x1024 訓練 損失
ax[0, 0].plot(epochs, acc, 'o-', label=label_train_acc, markersize=3, linewidth=1)
ax[0, 1].plot(epochs, loss, 'o-', label=label_train_loss , markersize=3, linewidth=1)

# 本研究 1024x1024 テスト 損失
ax[0, 0].plot(epochs, val_acc, 'x-', label=label_valid_acc, markersize=4, linewidth=1)
ax[0, 1].plot(epochs, val_loss, 'x-', label=label_valid_loss, markersize=4, linewidth=1)

        
ax[0, 0].legend(fontsize=12.5)         # 凡例を表示
ax[0, 1].legend(fontsize=12.5)         # 凡例を表示
fig.savefig(dst, dpi=600)