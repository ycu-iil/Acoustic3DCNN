# Acoustic3DCNN: 3D Convolution Neural Network-based Identification of 3D acoustic image of benthic organisms in shallow marine sediments

- 門井さんへ
  - dataとfigディレクトリを作成してください
  - level02の/data1/Acoustic3D_dataの中にある, 音響データControl Hydrobiaulvae Macoma Mixを上で作ったdataディレクトリにコピーしてください
  - python 3D_CNN_train_test_visualization.pyで全てが走ります. つまり, データ読み込み, 3クラス分類の5fold CVの訓練とテスト, Grad-CAMによる可視化が行われます.
  - 主なパラメータについては3D_CNN_train_test_visualization.pyのmainの最初の方を見てください. (ちゃんと設定ファイルを別で用意したり, 引数で指定できるように...してみてください...) 動くことを確認したら説明します. 

This is the implenetation of 3DCNN model for classification of benthic organisms in 3D acoustic images [1].

[1] ...


## Requirements
- python 3.8  
- tensorflow==2.5
- Keras 2.6.0
- Pillow 8.0
- matplotlib, more-itertools, scipy, pydot, graphviz

## How to Use
1. Clone this repository and move into it.

2. Load the dataset (...) and put them in the "data" directory. 

3. Training, evaluation, and Visualization are done by the following command.  

```bash
python 3D_CNN_train_test_visualization.py
```
