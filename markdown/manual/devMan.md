# 環境構築手順書

# 1 本書について

本書では、3D都市モデル生成シミュレータシステム（以下「本システム」という。）の利用環境構築手順について記載しています。

# 2 動作環境

本システムの動作環境は以下のとおりです。

| 項目     | 最小動作環境       | 推奨動作環境      |
|--------|--------------|-------------|
| OS     | Ubuntu 20.08 | 同左          |
| GPU    | メモリ12GB以上    | NVIDIA A100 |
| Python | Python==3.9  | 同左          |
| CUDA   | CUDA>=11.3   | CUDA==12.4  |


# 3 サーバー環境構築及びライブラリインストール手順

## Dockerのインストール

以下のコマンドを使ってインストールします。
```
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt update
sudo apt install docker-ce
```

## NVIDIAドライバのインストール

DockerコンテナでGPUを使用するためには、まずNVIDIAのGPUドライバがインストールされている必要があります。以下のコマンドでドライバをインストールします。
```
sudo apt install nvidia-driver-565
```

## NVIDIA Container Toolkitのインストール

GPUを使うために、NVIDIAのContainer Toolkitをインストールします。これにより、Dockerコンテナ内でGPUリソースを利用できるようになります。

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

# 4 準備物一覧

アプリケーションを利用するために以下のデータを入手します。

|     | データ種別      | 用途              | 形式         |
|-----|------------|-----------------|------------|
| ①   | コード及びモデル   | 3D都市モデル生成       | .zip       |
| ②   | Dockerfile | サーバー計算環境の構築     | Dockerfile |


データ準備完了後、下記のコマンドで.zipを解凍：
```
unzip bridge2025.zip
```

ファイル構造は以下の通りです：
```
root
├── bridge2025
└── Dockerfile
```

bridge2025は下記の機械学習と深層学習のモデルが含まれています：

|   | データ種別           | 用途         | 形式    |
|---|-----------------|------------|-------|
| ① | 深層学習建物自動抽出モデル   | 建物生成（仮想都市） | .pth  |
| ② | 深層学習建物屋根自動分類モデル | 建物生成（仮想都市） | .pth  |
| ③ | 機械学習建物高さ自動予測モデル | 建物生成（共通）   | .json |
| ④ | 深層学習植生自動抽出モデル   | 植生生成（実都市）  | .onnx |
| ⑤ | 深層学習道路自動抽出モデル   | 道路生成（実都市）  | .onnx  |
| ⑥ | 生成AI建物生成モデル    | 建物生成（実都市）  | .ckpt |

モデルの置く場所は以下の通りです：
```
bridge2025
├── Roof_classification_inference
│      ├──Building_extraction
│      │　　└──model
│      │　　　　└──model_best.pth ①
│      └──Roof_classification
│       　　└──model
│       　　　　└──best_model.pth ②
├── Para_calc
│      └──xgb_model_20250109-113703.json ③
├── bg_extract
│      ├──tensorrt_veg
│      │　　　　└──end2end.onnx ④
│      └──tensorrt_road
│       　　　　└──end2end.onnx ⑤
└── Building_Generation_Opening
       └──BldgXL
        　　　　└──plateau_lod2_type_mixed.pt ⑥
```


# 5 プログラム実行

```
docker build -t bridge2025 .
docker run --gpus all --name bridge -it bridge2025:latest /bin/bash
cd bridge2025
python main.py
```