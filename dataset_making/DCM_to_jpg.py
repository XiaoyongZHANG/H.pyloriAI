import os
import platform
from smb.SMBConnection import SMBConnection
import pydicom
from PIL import Image
import numpy as np

#####################################################################
# 
##   張研究室NASにアクセスし、DICOM→.jpg変換と.jpgのローカル保存を行う ##
#
#####################################################################

#########################################   オプション   ########################################################
# 資格情報
cred_info = {
    "username"     :  "t_chubachi",
    "password"     :  "t_chubachi_6129",
    "my_name"      :   platform.uname().node,   # 実行中のコンピュータのネットワーク名
    "remote_name"  :  "ZhangLabNAS-1",
    "domain"       :  "",   # 指定しない
    "ip"           :  "10.66.5.180",
    "port"         :  "139",  # TCPポート
    "service_name" :  "Lab_Data"  # 共有フォルダ名 Lab_Data もしくは LAB_Document
}

# データの格納場所
base = "Common/00_Experimental_Data/04_Stomach_X_Ray_for_Pyrori/2019_Miyagi_Cancer_Society"
labels = ["infected", "non-infected"]
################################################################################################################

class Converter:

    def __init__(self, cred_info):
        self.__username = cred_info["username"]
        self.__password = cred_info["password"]
        self.__my_name  = cred_info["my_name"]
        self.__remote_name = cred_info["remote_name"]
        self.__domain = cred_info["domain"]
        self.__ip = cred_info["ip"]
        self.__port = cred_info["port"]
        self.__service_name = cred_info["service_name"]

    def __connection_setup(self):
        conn = SMBConnection(
            username=self.__username,
            password=self.__password,
            my_name=self.__my_name,
            remote_name=self.__remote_name,
            domain=self.__domain,
            use_ntlm_v2=True
        )
        return conn
    
    def __connect(self, conn):
        return conn.connect(ip=self.__ip, port=self.__port)

    def __get_foldersPath(self, conn, path):

        items = conn.listPath(self.__service_name, path)

        folderPaths = []

        for item in (i for i in items if not i.filename in['.', '..']):

            # フォルダの場合 パスを追加
            if item.isDirectory:
                folderPaths.append(f'{path}/{item.filename}')
        
        return folderPaths

    def __get_filesPath(self, conn, path):

        items = conn.listPath(self.__service_name, path)

        filePaths = []

        for item in (i for i in items if not i.filename in['.', '..']):

            # ディレクトリではない場合
            if not item.isDirectory:
                #print("dir", f'{self.__service_name}/{path}/{item.filename}')
                filePaths.append(f'{path}/{item.filename}')
        
        return filePaths

    def __convert_dcm2jpg(self, path):
        # JPG変換
        ds = pydicom.dcmread(path)  # ds : bytesファイル

        arr = ds.pixel_array.astype(float) # Pixel Data を ndarray に変換

        arr_normalized = (arr / arr.max())*255
        arr_normalized = np.uint8(arr_normalized) # float to int
        img = Image.fromarray(arr_normalized, mode="L") # ndarray から画像を生成   L: 8bit グレースケール

        return img
    
    def __fix_aspect(self, img):
        # 横幅を基準に縦幅を調整する
        w, h = img.size
        if w < h:
            img = img.crop((0, int(h/2 - w/2), w, int(h/2 + w/2)))
        
        return img

    def dcm2jpg(self, labels, input_dir, output_dir):
        
        conn = self.__connection_setup()
        if self.__connect(conn):

            # ラベル毎に実行
            for label in labels:

                # フォルダのパスを取得
                folderPaths = self.__get_foldersPath(conn, input_dir[label])

                # 1患者ずつ処理を実行
                for i, folderPath in enumerate(folderPaths):

                    # 格納先ディレクトリ作成
                    os.makedirs(f'{output_dir}/{label}/{i}', exist_ok=True)

                    # 1患者のファイルのパスを取得
                    filePaths = self.__get_filesPath(conn, folderPath)

                    # 一時ファイルパス
                    temp_path = f'{output_dir}/temp.dcm'

                    for j, path in enumerate(filePaths):

                        # ファイルを一時ダウンロード
                        with open(temp_path, 'wb') as file:
                            conn.retrieveFile(self.__service_name, path, file)
                        
                        img = self.__convert_dcm2jpg(temp_path)
                        img = self.__fix_aspect(img)                        
                        #img = img.resize((1024, 1024))

                        img.save(f'{output_dir}/{label}/{i}/{j}.jpg')
                        os.remove(temp_path)
        
        conn.close()


if __name__ == "__main__":

    input_dir = {
        # Lab_Data からの path (ラベル数に応じて適宜追加 )
        labels[0] : os.path.join(base, "H.Pylori infected"),        # ラベル : 感染
        labels[1] : os.path.join(base, "H.Pylori noninfected")  # ラベル : 非感染
    }

    converter = Converter(cred_info)  # インスタンス化

    converter.dcm2jpg(labels, input_dir, output_dir = "/work/datasets/H.pylori_original")
