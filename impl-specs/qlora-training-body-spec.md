# QLoRA Training Body 設計メモ

## 目的
`sleep_consolidation` が生成した JSONL データを用いて、LoRA/QLoRA による学習を実行する。

## 位置づけ
- 記憶圧縮: `sleep_consolidation.py`
- 学習接続: `sleep_consolidation_training.py`
- 実学習本体: 本書

この段階は、入力検証やアダプタ保存ではなく、学習ループそのものを担う。

## 公開インターフェース

### `QLoRATrainer`
- 学習実行の薄いラッパーとする。
- 外部 trainer 実装を注入できるようにする。

### `prepare_model_for_training(model)`
- 学習可能なベースモデルを前処理する。
- 4bit 量子化や k-bit training 前処理を適用する。

### `train(examples, model, output_dir)`
- 学習例を受け取り、学習を実行する。
- 返り値は学習の要約情報とする。

### `save_adapter(output_dir)`
- 学習済みアダプタを保存する。

### `load_adapter(base_model, adapter_dir)`
- 保存済みアダプタを再読込する。

## 入力ルール
- 学習例は `input / thought / output` を持つ。
- `thought` は空でないこと。
- `status` は `confirmed` のみを学習対象とする。
- 低信頼・矛盾候補は事前に除外済みであること。

## 学習ルール
- 学習は causal language modeling を前提とする。
- `input` と `thought` を含む prompt から `output` を学習する。
- `thought` は内部推論の教師として扱う。
- 勾配更新は LoRA/QLoRA に限定する。

## モデル契約
- ベースモデルは学習可能な causal LM とする。
- 4bit ロードを許容する。
- `peft.prepare_model_for_kbit_training` 相当の前処理を適用できること。

## エラー時の挙動
- 無効な例はスキップする。
- 学習失敗時は中間成果物を破棄しない。
- 保存失敗時は学習済み状態を破棄せずに失敗する。

## 境界条件
- 空データセットは安全に終了する。
- 長すぎる例は事前分割済みであることを前提とする。
- `thought` が空の例は除外する。

## 仕様テスト観点
- 学習前処理が適用されること
- 学習例が正しい prompt 形式に変換されること
- 空データセットで安全に終了すること
- 学習後にアダプタが保存されること
- 保存済みアダプタを再読込できること

## 非目標
- 記憶圧縮
- JSONL 生成
- データ品質の評価指標
- 長期学習のドリフト制御

## 補足
- 本書は「学習の実行器」に限定し、データ生成と保存先管理は別モジュールに委譲する。
