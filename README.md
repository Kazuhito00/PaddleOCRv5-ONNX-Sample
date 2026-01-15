# PaddleOCRv5-ONNX-Sample
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)のPythonでのONNX推論サンプルです。<br>
PaddleOCR v5モデルを pyclipper、shapely 依存無しに ONNXRuntime で推論できるようにしたものです。<br>

# Model
PaddleOCR v5の以下のモデルを使用しています:
* PP-OCRv5_mobile_det_infer.onnx (テキスト検出モデル)
* PP-OCRv5_mobile_rec_infer.onnx (テキスト認識モデル)
* PP-OCRv5_server_det_infer.onnx (サーバー版検出モデル)
* PP-OCRv5_server_rec_infer.onnx (サーバー版認識モデル)

変換自体を試したい方はColaboratoryなどで[PaddleOCRv5-Convert2ONNX.ipynb](PaddleOCRv5-Convert2ONNX.ipynb)を使用ください。<br>

# Requirement
requirements.txt を参照ください。

必要なパッケージ:
* opencv-python
* Pillow
* numpy
* onnxruntime (GPU使用時は onnxruntime-gpu)

# Installation
```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

# Demo
デモ(シンプルなOCR)の実行方法は以下です。
```bash
python demo_simple_ocr_en.py --image=sample.jpg
```
* --image<br>
OCR対象画像の指定<br>
デフォルト：sample.jpg
* --det_model<br>
テキスト検出モデルの指定<br>
デフォルト：./ppocr_onnx/model/det_model/PP-OCRv5_mobile_det_infer.onnx
* --rec_model<br>
テキスト認識モデルの指定<br>
デフォルト：./ppocr_onnx/model/rec_model/PP-OCRv5_mobile_rec_infer.onnx
* --rec_char_dict<br>
辞書データの指定<br>
デフォルト：./ppocr_onnx/model/ppocrv5_dict.txt
* --use_gpu<br>
GPU推論の利用<br>
デフォルト：指定なし


デモ(日本語検出＋表示)の実行方法は以下です。
```bash
python demo_draw_ocr_ja.py --image=sample.jpg
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：640
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：360
* --det_model<br>
テキスト検出モデルの指定<br>
デフォルト：./ppocr_onnx/model/det_model/PP-OCRv5_mobile_det_infer.onnx
* --rec_model<br>
テキスト認識モデルの指定<br>
デフォルト：./ppocr_onnx/model/rec_model/PP-OCRv5_mobile_rec_infer.onnx
* --rec_char_dict<br>
辞書データの指定<br>
デフォルト：./ppocr_onnx/model/ppocrv5_dict.txt
* --use_gpu<br>
GPU推論の利用<br>
デフォルト：指定なし

# Architecture
このサンプルは以下の2段階のOCRパイプラインを実装しています:

1. **テキスト検出 (Text Detection)**: 画像からテキスト領域のバウンディングボックスを検出
2. **テキスト認識 (Text Recognition)**: 検出領域から実際のテキストを認識

メインクラス `PaddleOcrONNX` が両方のコンポーネントを統合し、以下の処理を行います:
* 検出ボックスを上から下、左から右の順にソート
* スコアフィルタリングで低信頼度の結果を除外
* 処理時間の測定

# Reference
* [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

# License
PaddleOCRv5-ONNX-Sample is under [Apache2.0 License](LICENSE).

# License(Font)
日本語フォントは[LINE Seed JP](https://seed.line.me/index_jp.html)を利用しています。
