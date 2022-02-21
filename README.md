## 概要
Nishika  AI×商標：イメージサーチコンペティション（類似商標画像の検出）  
2nd place solution  
Discussion：https://www.nishika.com/competitions/22/topics/213

## OSなど
- Ubuntu 18.04.6
- CUDA Version 11.2
- Python 3.7.12

<br>

## 必要なライブラリ
- Docker image : `gcr.io/kaggle-gpu-images/python:v107`
- pytorch-lightning
- pytorch-metric-learning
- timm
- augly
- addict
- rich
- faiss-cpu
```
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchtext==0.11.1 pytorch-lightning==1.5.8 pytorch-metric-learning==1.1.0 timm==0.5.4 augly==0.2.1 addict==2.4.0 rich==11.0.0 faiss-cpu==1.7.2
pip install -U scikit-learn
apt-get install python3-magic
```

<br>

## 乱数シードなどの設定値情報
- `config` フォルダの各yamlファイルに設定値を記載しています

<br>

## モデルの学習から予測まで行う際の、ソースコードの実行手順
1. `data`フォルダへ[competitionデータ](https://www.nishika.com/competitions/22/data)を保存（要zip解凍）
2. `python preprocess.py` (前処理)
3. `python train_convnext.py` (ConvNext学習)
4. `python train_swin.py` (Swin Transformer学習)
5. `settings.json`の**USE_PRETRAIN**を**False**に設定
6. `python make_submit.py` (Submitファイル作成)
