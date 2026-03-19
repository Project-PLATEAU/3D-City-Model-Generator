# 操作マニュアル

## 本書について

本書では、3D都市モデル生成シミュレータシステム（以下「本システム」という。）の操作手順について記載しています。

### 3D都市モデル生成までの流れ

![total_workflow](../resources/fig04.png)

## 1 LOD1モデル生成

デジタルシティサービスにアクセスし、生成を行う画面へ移動する。 
http://localhost:8080

![func_selection](../resources/fig09.png)

①建物フットプリントをアップロードし、「生成」をクリックすると、 
②建物フットプリントデータ範囲内のLOD1モデルが生成・可視化される。

![lod1_gen](../resources/fig05.png)

## 2 LOD2モデル生成

①衛星画像をアップロードし、「生成」をクリックすると、
②範囲内のLOD2モデルが生成・可視化される。

![lod2_gen](../resources/fig06.png)

## 3 LOD3モデル生成

①街路画像とMMS点群をアップロードし、「生成」をクリックすると、
②範囲内のLOD3モデルが生成・可視化される。

![lod3_gen](../resources/fig07.png)

## 4 可視化するレイヤの選択

表示したい生成結果のレイヤを選択（LOD別、BMQI可視化など）する。

![layer_setting](../resources/fig08.png)



