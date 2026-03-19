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

## データとモデル準備

本システムでは、大容量の深層学習モデルファイルを多数扱うため、すべてのモデルファイルをGitHubに含めることができません。したがって、以下のURLより、あらかじめ構成されたモデルファイル一式をダウンロードする必要があります。  
[モデルファイルのダウンロード](https://www.geospatial.jp/ckan/dataset/3daiready)

## python環境の構築

以下のコマンドを使ってインストールします。
```
cd Bridge2025UI
conda env create -f environment.yml
conda activate gen3d_UI_2025
pip install -r requirements.txt
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -U "urllib3==1.26.18"
```

## フロントエンド環境の構築

本システムではフロントエンドの実行環境としてNode.jsを使用します。Node.jsのバージョン管理のためにnvm（Node Version Manager）を用いて、指定バージョンをインストールし、使用環境を構築します。
```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.zshrc
nvm install 18
nvm use 18
```

外部ネットワークへの接続やトンネル通信のためにcloudflaredを使用します。以下のコマンドにより、公式リリースからインストーラ（.debファイル）をダウンロードし、システムにインストールします。
```
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```


# 4 準備物一覧

アプリケーションを利用するために以下のデータを入手します。

|     | データ種別      | 用途              | 形式         |
|-----|------------|-----------------|------------|
| ①   | コード及びモデル   | 3D都市モデル生成       | .zip       |


データ準備完了後、下記のコマンドで.zipを解凍：
```
unzip Bridge2025UI.zip
```

ファイル構造は以下の通りです：
```
root
├── BldgGen2025
├── BridgeUI
├── environment.yml
└── requirements.txt
```

Bridge2025UIは下記の機械学習と深層学習のモデルが含まれています：

|   | データ種別               | 用途         | 形式    |
|---|---------------------|------------|-------|
| ① | 生成AI建物生成モデル         | 建物生成   | .ckpt |
| ② | 深層学習植生自動抽出モデル       | 植生生成   | .onnx |
| ③ | 深層学習MMS点群開口部自動抽出モデル | LOD3生成    | .pt   |
| ④ | 深層学習街路画像開口部自動抽出モデル  | LOD3生成 | .pt  |

モデルの置く場所は既にBridge2025UI.zipで配置完了しています：

# 5 プログラム実行

ツールを起動するには、まずフロントエンド環境を構築します。次に、既存の依存関係を一度削除してから再インストールを行い、その後ビルドを実行し、開発用サーバーを起動します。

```
conda activate gen3d_UI_2025
cd BridgeUI
rm -rf node_modules package-lock.json
npm install
npm run build
npm run dev:full
```

ローカルサーバーを外部公開するため、以下のコマンドを実行します。
```
cloudflared tunnel --url http://localhost:8080 --protocol http2
```

# 6 本ツールを利用するにあたりユーザが準備する入力データ

本ツールを用いて 3D 都市モデルを生成するためには、以下の入力データをユーザ側で準備する必要があります。

- 衛星画像データ  
- 建築物フットプリントデータ  
- MMS（Mobile Mapping System）データ  

各データの仕様を以下に示します。

---

## 6.1 衛星画像データ

### （1）概要

衛星画像は、建築物モデル生成における基礎データとなります。

### （2）推奨仕様

| 項目 | 推奨条件 |
|------|----------|
| 解像度 | 0.3m 以上 |
| バンド | RGB（マルチスペクトル対応可） |
| 投影座標系 | 平面直角座標系 または UTM |
| データ形式 | GeoTIFF |
| 幾何補正 | オルソ補正済み |

---

## 6.2 建築物フットプリントデータ

### （1）概要

建築物のフットプリントは、3D 都市モデル生成の基礎となるポリゴンデータです。

### （2）推奨仕様

| 項目 | 推奨条件 |
|------|----------|
| データ形式 | GeoJSON |
| ジオメトリ | ポリゴン |
| トポロジ | 自己交差なし |
| 座標系 | 衛星画像と同一 |

---

## 6.3 MMS（Mobile Mapping System）データ

### （1）概要

MMS データは、建築物の LOD3 モデル生成時の開口部（窓・扉）、都市設備、植生抽出に利用します。

### （2）利用可能データ種別

- 沿道画像（全周画像または前方画像）  
- 三次元点群（LiDAR）  
- 位置情報（IMU）

### （3）推奨仕様

| 項目 | 推奨条件 |
|------|----------|
| 点群密度 | 500 点 / m² 以上推奨 |
| 画像解像度 | 4K 相当以上 |
| 位置精度 | 水平誤差 10cm 以下 |
| データ形式 | LAS / LAZ / JPEG / PNG |

---