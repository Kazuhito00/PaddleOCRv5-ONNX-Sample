"""
PaddleOCR v5 ONNX推論メインクラス
"""
import os
import copy
import time
import math

import cv2
import numpy as np
import onnxruntime


class PaddleOcrONNX(object):
    def __init__(self, args):
        """
        PaddleOCR v5 ONNX推論の初期化

        Args:
            args: 設定パラメータ
        """
        self.args = args

        # ONNXRuntimeのプロバイダ設定
        if args.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # 検出モデルの読み込み
        print(f"Loading detection model: {args.det_model_dir}")
        self.det_session = onnxruntime.InferenceSession(
            args.det_model_dir, providers=providers
        )

        # 認識モデルの読み込み
        print(f"Loading recognition model: {args.rec_model_dir}")
        self.rec_session = onnxruntime.InferenceSession(
            args.rec_model_dir, providers=providers
        )

        # 文字辞書の読み込み
        print(f"Loading dictionary: {args.rec_char_dict_path}")
        self.char_dict = self._load_dict(args.rec_char_dict_path)

        # パラメータ設定
        self.det_limit_side_len = args.det_limit_side_len
        self.det_limit_type = args.det_limit_type
        self.det_db_thresh = args.det_db_thresh
        self.det_db_box_thresh = args.det_db_box_thresh
        self.det_db_unclip_ratio = args.det_db_unclip_ratio

        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.drop_score = args.drop_score

        print("Model loaded successfully!")

        self.crop_image_res_index = 0

    def _load_dict(self, dict_path):
        """文字辞書を読み込む"""
        with open(dict_path, "r", encoding="utf-8") as f:
            chars = [line.strip() for line in f]
        chars = ["blank"] + chars
        return chars

    def _resize_for_det(self, img):
        """検出用に画像をリサイズ"""
        h, w = img.shape[:2]

        # 小さすぎる画像のパディング
        if h + w < 64:
            pad_h = max(32, h)
            pad_w = max(32, w)
            padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
            padded[:h, :w, :] = img
            img = padded
            h, w = img.shape[:2]

        # リサイズ比率の計算
        if self.det_limit_type == "max":
            if max(h, w) > self.det_limit_side_len:
                ratio = float(self.det_limit_side_len) / max(h, w)
            else:
                ratio = 1.0
        elif self.det_limit_type == "min":
            if min(h, w) < self.det_limit_side_len:
                ratio = float(self.det_limit_side_len) / min(h, w)
            else:
                ratio = 1.0
        else:
            ratio = float(self.det_limit_side_len) / max(h, w)

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # 32の倍数に調整
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        resized = cv2.resize(img, (resize_w, resize_h))

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return resized, (h, w, ratio_h, ratio_w)

    def _normalize_image(self, img):
        """画像の正規化（検出用）"""
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return img

    def _preprocess_det(self, img):
        """検出モデル用の前処理"""
        resized, shape_info = self._resize_for_det(img)
        normalized = self._normalize_image(resized)

        # HWC -> CHW
        transposed = normalized.transpose(2, 0, 1)

        # バッチ次元を追加
        batched = np.expand_dims(transposed, axis=0).astype(np.float32)

        return batched, shape_info

    def _unclip(self, box, unclip_ratio=1.5):
        """検出ボックスを拡張"""
        box = np.array(box, dtype=np.float32)

        if len(box) < 3:
            return box.reshape(1, -1, 2)

        # 最小外接矩形を取得
        rect = cv2.minAreaRect(box.reshape(-1, 1, 2))
        center, (width, height), angle = rect

        # 幅と高さが0の場合は元のボックスを返す
        if width < 1e-6 or height < 1e-6:
            return box.reshape(1, -1, 2)

        # 面積と周長を計算
        area = width * height
        perimeter = 2 * (width + height)

        # オフセット距離を計算
        distance = area * unclip_ratio / perimeter
        distance *= 1.10

        # 各辺をdistance分だけ外側に移動
        new_width = width + 2 * distance
        new_height = height + 2 * distance

        # 拡張した矩形を再構築
        new_rect = (center, (new_width, new_height), angle)
        expanded = cv2.boxPoints(new_rect)

        return expanded.astype(np.float32).reshape(1, -1, 2)

    def _box_score_fast(self, bitmap, _box):
        """高速なボックススコア計算"""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def _get_mini_boxes(self, contour):
        """輪郭から最小外接矩形を取得"""
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0

        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return np.array(box), min(bounding_box[1])

    def _postprocess_det(self, pred, shape_info):
        """検出結果の後処理"""
        ori_h, ori_w, ratio_h, ratio_w = shape_info
        pred = pred[0, 0]

        height, width = pred.shape

        # 二値化
        segmentation = pred > self.det_db_thresh
        mask = (segmentation * 255).astype(np.uint8)

        # 輪郭検出
        outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            _, contours, _ = outs
        else:
            contours, _ = outs

        num_contours = min(len(contours), 1000)

        boxes = []
        scores = []

        for index in range(num_contours):
            contour = contours[index]

            # 最小外接矩形
            points, sside = self._get_mini_boxes(contour)
            if sside < 3:
                continue
            points = np.array(points)

            # スコア計算
            score = self._box_score_fast(pred, points.reshape(-1, 2))
            if self.det_db_box_thresh > score:
                continue

            # ボックスを拡張
            box = self._unclip(points, self.det_db_unclip_ratio)
            if box is None or len(box) == 0:
                continue
            box = box.reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < 5:
                continue
            box = np.array(box)

            # 元のサイズにスケール
            box[:, 0] = np.clip(np.round(box[:, 0] / width * ori_w), 0, ori_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * ori_h), 0, ori_h)

            boxes.append(box.astype("int32"))
            scores.append(score)

        return boxes, scores

    def _detect(self, img):
        """テキスト領域を検出"""
        start_time = time.time()

        input_data, shape_info = self._preprocess_det(img)

        input_name = self.det_session.get_inputs()[0].name
        output = self.det_session.run(None, {input_name: input_data})[0]

        boxes, scores = self._postprocess_det(output, shape_info)

        elapse = time.time() - start_time

        return boxes, elapse

    def _resize_norm_img_rec(self, img, max_wh_ratio=None):
        """認識用に画像をリサイズして正規化"""
        img_channel = 3
        img_height = self.rec_image_shape[1]
        img_width = self.rec_image_shape[2]

        if max_wh_ratio is not None:
            img_width = int(img_height * max_wh_ratio)

        h, w = img.shape[:2]
        ratio = w / float(h)

        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))

        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype("float32")

        # 正規化: (x / 255 - 0.5) / 0.5
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        # パディング
        padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def _preprocess_rec_batch(self, img_list):
        """認識モデル用のバッチ前処理"""
        img_height = self.rec_image_shape[1]
        img_width = self.rec_image_shape[2]

        # 各画像のアスペクト比を計算
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # 最大アスペクト比を計算
        max_wh_ratio = img_width / img_height
        for w_ratio in width_list:
            max_wh_ratio = max(max_wh_ratio, w_ratio)

        # バッチを作成
        norm_img_batch = []
        for img in img_list:
            norm_img = self._resize_norm_img_rec(img, max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch)
        return norm_img_batch.astype(np.float32)

    def _postprocess_rec(self, pred):
        """認識結果の後処理（CTCデコード）"""
        preds_idx = pred.argmax(axis=2)
        preds_prob = pred.max(axis=2)

        results = []
        batch_size = len(preds_idx)

        for batch_idx in range(batch_size):
            text_index = preds_idx[batch_idx]
            text_prob = preds_prob[batch_idx]

            char_list = []
            conf_list = []

            selection = np.ones(len(text_index), dtype=bool)
            selection[1:] = text_index[1:] != text_index[:-1]
            selection &= text_index != 0

            for i, selected in enumerate(selection):
                if selected and text_index[i] < len(self.char_dict):
                    char_list.append(self.char_dict[text_index[i]])
                    conf_list.append(text_prob[i])

            text = "".join(char_list)
            confidence = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
            results.append((text, confidence))

        return results

    def _recognize(self, img_list):
        """テキストを認識"""
        start_time = time.time()

        input_data = self._preprocess_rec_batch(img_list)

        input_name = self.rec_session.get_inputs()[0].name
        output = self.rec_session.run(None, {input_name: input_data})[0]

        results = self._postprocess_rec(output)

        elapse = time.time() - start_time

        return results, elapse

    def _get_rotate_crop_image(self, img, points):
        """検出ボックスから画像を切り出して回転補正"""
        points = np.array(points, dtype=np.float32)

        width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])
            )
        )
        height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])
            )
        )

        dst_points = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )

        M = cv2.getPerspectiveTransform(points, dst_points)
        cropped = cv2.warpPerspective(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )

        if height > width * 1.5:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

        return cropped

    def sorted_boxes(self, dt_boxes):
        """
        テキストボックスを上から下、左から右の順にソート
        """
        if len(dt_boxes) == 0:
            return []

        boxes_with_y = [(box, np.mean(box[:, 1])) for box in dt_boxes]
        boxes_with_y.sort(key=lambda x: x[1])

        return [box for box, _ in boxes_with_y]

    def __call__(self, img):
        """
        OCR実行（検出 + 認識）

        Args:
            img: 入力画像

        Returns:
            dt_boxes: 検出ボックスのリスト
            rec_res: 認識結果のリスト [(text, score), ...]
            time_dict: 処理時間の辞書
        """
        time_dict = {'det': 0, 'rec': 0, 'all': 0}
        start = time.time()

        ori_im = img.copy()

        # テキスト検出
        dt_boxes, elapse = self._detect(img)
        time_dict['det'] = elapse

        if dt_boxes is None or len(dt_boxes) == 0:
            return None, None, time_dict

        # ボックスをソート
        dt_boxes = self.sorted_boxes(dt_boxes)

        # 検出領域を切り出し
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self._get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # テキスト認識
        rec_res, elapse = self._recognize(img_crop_list)
        time_dict['rec'] = elapse

        # スコアフィルタリング
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        end = time.time()
        time_dict['all'] = end - start

        return filter_boxes, filter_rec_res, time_dict
