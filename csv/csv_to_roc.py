##################################################################
# 
##   .csv からk分割交差検証における各ROC曲線と平均AUCをプロットする ##
#
##################################################################

#########################################   オプション   ########################################################
base = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10'
lang_jp = True    # 日本語で書き込み
################################################################################################################

import matplotlib.pyplot as plt
from sklearn import metrics
import csv
import os
import numpy as np
import japanize_matplotlib

dir_path = os.path.join(base, 'csv/tpr_fpr')
dst = os.path.join(base, 'graph/roc/mean_roc.jpg')
os.makedirs(os.path.join(base, 'graph/roc'), exist_ok=True)

fig, ax = plt.subplots(1, 1, tight_layout=True, squeeze=False) # squeeze : axesを常に2次元配列として扱う

fprs = []
tprs = []
aucs = []

# 作成したモデル分繰り返す
for i, path in enumerate(os.listdir(dir_path)):
    
    path = os.path.join(dir_path, path)
    
    with open(path) as f:
        reader = csv.reader(f, delimiter=',')
        l = [row for row in reader]
        fpr = [float(s) for s in l[0][1:]]
        tpr = [float(s) for s in l[1][1:]]
        threshold = [float(s) for s in l[2][1:]]

        # AUCの計算
        fprs.extend(fpr)
        tprs.extend(tpr)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)
        
        youden_index = np.argmax(np.array(tpr)-np.array(fpr))
    
        #print(threshold[youden_index], tprs[youden_index], fprs[youden_index])

        # プロット
        #ax[0, 0].plot(fpr, tpr, "-^", label="ROC fold "+str(i)+": AUC = %.3f "%auc + "\nYouden Index = %.3f"%threshold[youden_index], markersize=2, alpha=0.1)  # - : 実線,  o : 丸,  ^ : 三角
        #ax[0, 0].plot(fpr[youden_index], tpr[youden_index], "-^", markersize=4)  # - : 実線,  o : 丸,  ^ : 三角
        ax[0, 0].plot(fpr, tpr, "-^", label="ROC fold"+str(i)+": AUC=%.4f "%auc, markersize=2, alpha=0.2)  # - : 実線,  o : 丸,  ^ : 三角

fprs.sort()
tprs.sort()
auc = metrics.auc(fprs, tprs)

# Mean ROC のプロット
if lang_jp:
   ax[0, 0].plot(fprs[::4], tprs[::4], "-o", label="AUC 平均　　: %.4f "%auc + "\nAUC 標準偏差: %.4f "%np.std(aucs), markersize=3, linewidth=1.0, color='red')  # - : 実線,  o : 丸,  ^ : 三角
else:
    ax[0, 0].plot(fprs[::4], tprs[::4], "-o", label="ROC Avg: AUC = %.4f "%auc + "\nAUC Std: %.4f "%np.std(aucs), markersize=5, linewidth=2.0, color='red')  # - : 実線,  o : 丸,  ^ : 三角

# 軸の設定
if lang_jp:
    ax[0, 0].set_xlabel("偽陽性率[%]", size=14)
    ax[0, 0].set_ylabel("真\n陽\n性\n率\n[%]", size=14, labelpad=15, rotation=0, va="center")

else:
    ax[0, 0].set_xlabel("False Positive Rate[%]", size=14)
    ax[0, 0].set_ylabel("True Positive Rate[%]", size=14)

ax[0, 0].set_xticks(np.arange(0, 1.1, step=0.1))
ax[0, 0].set_yticks(np.arange(0, 1.1, step=0.1))
ax[0, 0].set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax[0, 0].set_yticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax[0, 0].grid()                         # 目盛り線を表示
ax[0, 0].legend(fontsize=12.5)          # 凡例を表示

fig.savefig(dst, dpi=300)