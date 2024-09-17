################################################################
# 
##   ローカルに保存されたオリジナルデータセットを分割数kで分割する ##
#
################################################################

#########################################   オプション   ########################################################
DIV_NUM = 10         # 2, 4, 5, 10, 20. 25, 50, 100
################################################################################################################

import os, shutil, random 

# K-分割交差検証
# データのシャッフル
# infected ラベル
n = list(range(100))
random.shuffle(n)
div_infected = []
for i in range(0, 100, int(100/DIV_NUM)):
    div_infected.append(n[i:i+int(100/DIV_NUM)])

# non-infected ラベル
random.shuffle(n)
div_non_infected = []
for i in range(0, 100, int(100/DIV_NUM)):
    div_non_infected.append(n[i:i+int(100/DIV_NUM)])

# データを分割する
train_infected = []
train_non_infected = []
validation_infected = []
validation_non_infected = []

for i in range(DIV_NUM):
    train_infected.append([])
    train_non_infected.append([])
    validation_infected.append(div_infected[i])
    validation_non_infected.append(div_non_infected[i])

    for j in range(DIV_NUM):
        if not i==j:
            train_infected[i].extend(div_infected[j])
            train_non_infected[i].extend(div_non_infected[j])



original_infected_dir = '/work/datasets/H.pylori original/H.pylori/infected'
original_non_infected_dir = '/work/datasets/H.pylori original/H.pylori/non-infected'

base_dir = '/work/datasets/H.pylori K-Fold/div' + str(DIV_NUM)

train_infected_dir = []
train_non_infected_dir = []
validation_infected_dir = []
validation_non_infected_dir = []

for i in range(DIV_NUM):

    div_dir = os.path.join(base_dir, str(i))

    train_dir = os.path.join(div_dir, 'train')
    validation_dir = os.path.join(div_dir, 'validation')

    train_infected_dir.append(os.path.join(train_dir, 'infected'))
    validation_infected_dir.append(os.path.join(validation_dir, 'infected'))

    os.makedirs(train_infected_dir[i], exist_ok=True)
    os.makedirs(validation_infected_dir[i], exist_ok=True)

    train_non_infected_dir.append(os.path.join(train_dir, 'non-infected'))
    validation_non_infected_dir.append(os.path.join(validation_dir, 'non-infected'))

    os.makedirs(train_non_infected_dir[i], exist_ok=True)
    os.makedirs(validation_non_infected_dir[i], exist_ok=True)


# データのコピー
# --non-infected--
# train
for i in range(DIV_NUM):
    fnames = []
    for j in range(len(train_non_infected[i])):    # 100/DIV_NUM 患者分
        for k in range(8):  # 8体位分
            fnames.append(str(train_non_infected[i][j]) + "/" + str(k) + ".jpg")

        # 格納先ディレクトリを作成
        dir = os.path.join(train_non_infected_dir[i], str(train_non_infected[i][j]))
        os.makedirs(dir, exist_ok=True)


    for fname in fnames:
        src = os.path.join(original_non_infected_dir, fname)
        dst = os.path.join(train_non_infected_dir[i], fname)
        print(dst)
        shutil.copyfile(src, dst)


# validation
for i in range(DIV_NUM):
    fnames = []
    for j in range(len(validation_non_infected[i])):    # 100/DIV_NUM 患者分
        for k in range(8):  # 8体位分
            fnames.append(str(validation_non_infected[i][j]) + "/" + str(k) + ".jpg")

        # 格納先ディレクトリを作成
        dir = os.path.join(validation_non_infected_dir[i], str(validation_non_infected[i][j]))
        os.makedirs(dir, exist_ok=True)


    for fname in fnames:
        src = os.path.join(original_non_infected_dir, fname)
        dst = os.path.join(validation_non_infected_dir[i], fname)
        print(dst)
        shutil.copyfile(src, dst)



# --infected--
# train
for i in range(DIV_NUM):
    fnames = []
    for j in range(len(train_infected[i])):    # 100/DIV_NUM 患者分
        for k in range(8):  # 8体位分
            fnames.append(str(train_infected[i][j]) + "/" + str(k) + ".jpg")

        # 格納先ディレクトリを作成
        dir = os.path.join(train_infected_dir[i], str(train_infected[i][j]))
        os.makedirs(dir, exist_ok=True)


    for fname in fnames:
        src = os.path.join(original_infected_dir, fname)
        dst = os.path.join(train_infected_dir[i], fname)
        print(dst)
        shutil.copyfile(src, dst)


# validation
for i in range(DIV_NUM):
    fnames = []
    for j in range(len(validation_infected[i])):    # 100/DIV_NUM 患者分
        for k in range(8):  # 8体位分
            fnames.append(str(validation_infected[i][j]) + "/" + str(k) + ".jpg")

        # 格納先ディレクトリを作成
        dir = os.path.join(validation_infected_dir[i], str(validation_infected[i][j]))
        os.makedirs(dir, exist_ok=True)


    for fname in fnames:
        src = os.path.join(original_infected_dir, fname)
        dst = os.path.join(validation_infected_dir[i], fname)
        print(dst)
        shutil.copyfile(src, dst)
