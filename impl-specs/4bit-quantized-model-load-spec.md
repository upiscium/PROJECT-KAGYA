# 4Bit Quantized Model Load 設計メモ

## 目的
QLoRA 学習の前提として、ベースモデルを 4bit 量子化で安定して読み込む。

## 位置づけ
- 記憶圧縮: `sleep_consolidation.py`
- JSONL 生成: `sleep_consolidation.py`
- 学習接続: `sleep_consolidation_training.py`
- 学習本体: `qlora_training.py`
- 4bit ロード: 本書

この段階は、学習前のモデルロードだけを担う。

## 公開インターフェース

### `QuantizedModelLoader`
- 4bit モデルロードの薄いラッパーとする。

### `load_4bit_model(model_name_or_path, device_map=None)`
- ベースモデルを 4bit で読み込む。
- 返り値はロード済みモデルとする。

### `build_quantization_config()`
- 量子化設定を明示的に組み立てる。

### `prepare_for_training(model)`
- 学習前処理を適用する。

## ロードルール
- 量子化設定は明示する
- 4bit ロード失敗時は例外を返すか、明示的にフォールバックする
- 学習前処理とセットで扱う
- GPU / CUDA 環境を前提とする

## モデル契約
- `transformers` の causal LM を対象とする
- `bitsandbytes` の 4bit ロードを利用する
- `peft.prepare_model_for_kbit_training` 相当の前処理を組み合わせる

## エラー時の挙動
- `bitsandbytes` が使えない場合は明示的に失敗する
- CUDA がない場合は安全に失敗する
- ロード失敗時に不完全なモデルを返さない

## 境界条件
- ロード対象が空なら失敗する
- 量子化設定が未指定でもデフォルトを持つ
- 実環境でのみ利用されることを前提とする

## 仕様テスト観点
- 量子化設定が構築されること
- ロード失敗時に明示的に失敗すること
- 学習前処理と接続できること
- フォールバック経路があること

## 非目標
- 学習ループ
- LoRA アダプタ保存
- 記憶圧縮
- JSONL 生成

## 補足
- この段階は、4bit ロードを単独で責務化し、後段の QLoRA 学習へつなぐために使う。
