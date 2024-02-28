# 環境構築手順書

# 1 本書について

本書では、3D都市モデル生成シミュレータシステム（以下「本システム」という。）の利用環境構築手順について記載しています。本システムの構成や仕様の詳細については以下も参考にしてください。

[技術検証レポート](https://www.mlit.go.jp/plateau/file/libraries/doc/plateau_tech_doc_0030_ver01.pdf)

# 2 動作環境

本システムの動作環境は以下のとおりです。

| 項目     | 最小動作環境       | 推奨動作環境      | 
|--------|--------------|-------------| 
| OS     | Ubuntu 20.08 | 同左          | 
| GPU    | メモリ16GB以上    | NVIDIA A100 | 
| Python | Anaconda     | 同左          | 
| CUDA   | CUDA>=10.2     | CUDA==12.1  | 


# 3 環境構築及びライブラリインストール手順

##GPU計算環境構築
公式チュートリアルを従ってCUDA10.2以上及びCUDNN7以上をインストール
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

##Anacondaインストール

https://www.anaconda.com/download

対応するシステムのバージョンをダウンロードした後、指示に従って直接インストール

Anacondaを環境変数として設定
```
echo 'export PATH="/your/anaconda3/path/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 仮想環境構築
```
conda env create -f environment.yaml
conda activate gen3d
```

## 深層学習ライブラリインストール
```
conda install -c conda-forge cudatoolkit-dev=11.3 -y
conda install -c　yacs -y
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install mmdet==2.28.1
```

## CUDAオペレータをコンパイルする
```
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage/ops_dcnv3
sh ./make.sh
```


# 4 準備物一覧

アプリケーションを利用するために以下のデータを入手します。

|     | データ種別                    | 用途         | 形式       |
|-----|--------------------------|------------|----------|
| ①   | 地形テンプレートとしてのDEMデータ       | 地形生成（共通）   | .tiff    |
| ②   | 建物配置テンプレートとしてのフットプリントデータ | 建物生成（仮想都市） | .geojson |
| ③   | 道路配置テンプレートデータ            | 道路生成（仮想都市） | .geojson |
| ④   | 設備3Dモデルテンプレートデータ         | 設備生成（共通）   | .obj     |
| ⑤   | 植生3Dモデルテンプレートデータ         | 植生生成（共通）   | .obj     |
| ⑥   | 建物3Dモデルテンプレートデータ         | 建物生成（仮想都市） | .obj     |
| ⑦   | 建物3Dモデル屋根タイプラベリングデータ     | 建物生成（仮想都市） | .csv     |
| ⑧   | 植生3Dモデル高低タイプラベリングデータ     | 植生生成（共通）   | .csv     |
| ⑨   | 深層学習植生自動抽出モデル            | 植生生成（実都市）  | .pth     |
| ⑩   | 深層学習道路自動抽出モデル            | 道路生成（実都市）  | .pth     |
| ⑪   | 生成式AI建物生成モデル             | 建物生成（実都市）  | .ckpt    |

データを準備した後、ファイル構造は以下のように：

仮想都市
```
gen3D_virtualCity
└── data
    ├── src_2d
    │      ├──dem　①
    │      └──shp　②③
    └── src_3d
           ├──lod3frn　④
           ├──lod3veg　⑤
           ├──obj　⑥
           ├──merged_filter1.csv　⑦
           └──tree_label.csv　⑧
```

実都市
```
gen3D_realCity
├── bg_extract
│   ├── ckpt ⑨⑩
│   └── ...
├── lightning_logs
│   ├──plateau_dataEnhancement_type1 ⑪
│   ├──plateau_dataEnhancement_type2 ⑪
│   ├──plateau_dataEnhancement_type3 ⑪
│   ├──plateau_dataEnhancement_NType5 ⑪
│   └──plateau_dataEnhancement_NType6 ⑪
└── ...
```


# 5 プログラム実行

##仮想都市
```
python gen_mesh.py 1024 --lod_building 2 --prob_t1 0.2 --prob_t2 0.3 --prob_t3 0.3 --prob_t4 0 --prob_t5 0.2 --prob_t6 0 --prob_t7 0 --lod_road 1 --road_width_main 1 --road_width_sub 0.1 --lod_vegetation 2 --low_tree_ratio 0.1 --high_tree_ratio 1 --lod_device 2 --telegraph_pole_ratio 1 --traffic_light_ratio 0.1 --lod_relief 1 --output ./result
```

##実都市
```
python gen3d.py --input gen3d_realCity_testData/mapbox/test02/footprint/footprint_test_2_selected.geojson --img gen3d_realCity_testData/mapbox/test02/satellite_image/test02_0_3.tiff --building_lod 2 --road_width 2 2
```