################################################################
# 
##             論文用 Drop%の分布を箱ひげ図に出力する            ##
#
################################################################

#########################################   オプション   ########################################################
src_dir = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10/csv/drop_gi_bi'
dst = '/work/output/pylori/VGG16_GAP/div4/1024x1024_batchsize16_epochs400/2023-10-19-15-08-10/csv/drop_gi_bi'
################################################################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import japanize_matplotlib
import seaborn as sns

path_infected = os.path.join(src_dir, 'infected_Drop_GI_BI.csv')
path_non_infected = os.path.join(src_dir, 'non-infected_Drop_GI_BI.csv')

sns.set()
sns.set_style('whitegrid')
sns.set_palette('Set3')

japanize_matplotlib.japanize()
     

# CSVファイルの読み込み
data = pd.read_csv(path_infected, header=None).T
data.columns = data.iloc[0]
data = data[1:]
drop_infected = np.array(data['Drop'].values, dtype='float32')

data = pd.read_csv(path_non_infected, header=None).T
data.columns = data.iloc[0]
data = data[1:]
drop_non_infected = np.array(data['Drop'].values, dtype='float32')

df = pd.DataFrame({
    'ピロリ菌感染': drop_infected,
    'ピロリ菌非感染': drop_non_infected
})
df_melt = pd.melt(df)

fig = plt.figure(figsize=(6, 7))
ax = fig.add_subplot(1, 1, 1)
sns.boxplot(x='variable', y='value', data=df_melt, showfliers=False, ax=ax)
sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Drop [%]')
ax.set_ylim(-1, 100)
fig.savefig(os.path.join(dst, "hakohige.jpg"), dpi=600)
