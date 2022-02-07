`python preprocess.py`

- RAW_DATA_DIRからtrain.csv, apply_images, cite_imagesを読み取る（settings.jsonで指定）
- 前処理を実行
- 前処理結果をPROCESSED_DATA_DIR（settings.jsonで指定）に保存

<br>

`python train_convnext.py`

- RAW_DATA_DIR, PROCESSED_DATA_DIRからデータを読み取る（settings.jsonで指定）
- configフォルダのconfig_train_convnext.yamlファイルから学習設定を取得
- ConvNextモデルをトレーニング
- train, citeデータのEmbedding作成
- モデルcheckpoint, EmbeddingをMODEL_CHECKPOINT_DIR/convnextに保存（settings.jsonで指定）

<br>

`python train_swin.py`

- RAW_DATA_DIR, PROCESSED_DATA_DIRからデータを読み取る（settings.jsonで指定）
- configフォルダのconfig_train_swin.yamlファイルから学習設定を取得
- Swin Transformerモデルをトレーニング
- train, citeデータのEmbedding作成
- モデルcheckpoint, EmbeddingをMODEL_CHECKPOINT_DIR/swinに保存（settings.jsonで指定）

<br>

`python make_submit.py`

- RAW_DATA_DIRからテストデータを読み取る（settings.jsonで指定）
- USE_PRETRAINがTrueの場合はPRETRAINED_MODEL_CHECKPOINT_DIR、  
Falseの場合はMODEL_CHECKPOINT_DIRからモデルをロード（settings.jsonで指定）
- SUBMISSION_DEVICE 上で推論処理実施（settings.jsonで指定）
- 処理は1画像ずつ前処理～推論を実行
- 予測結果をSUBMISSION_DIR（settings.jsonで指定）に保存
