# Sleep Consolidation Training Pipeline 設計書

## 目的
`sleep_consolidation.py` が生成した `JSONL` データを用いて学習を実行し、LoRA アダプタを保存・再読込できる後段パイプラインを定義する。

## 位置づけ
- 記憶圧縮: `sleep_consolidation.py`
- データ生成: `sleep_consolidation.py`
- 学習接続: 本書

この段階は、データ生成の後処理として扱う。記憶圧縮と学習は責務を分離する。

## 公開インターフェース

### `train(dataset_path, model, output_dir)`
- `JSONL` を読み込み、学習を実行する。
- 返り値は学習結果の要約情報とする。

### `save_adapter(output_dir)`
- 学習済み LoRA アダプタを保存する。
- 保存先は指定ディレクトリとする。

### `load_adapter(base_model, adapter_dir)`
- 保存済みアダプタをベースモデルへアタッチする。
- 起動時の復元処理として扱う。

## 入出力

### 入力
- `dataset_path`: 学習用 JSONL のパス
- `model`: 学習可能な causal language model
- `output_dir`: 学習結果の保存先

### 出力
- 学習件数
- 失敗件数
- 保存先パス
- ロード結果

## データ契約
学習データは各行が次のスキーマを満たす。

```json
{
  "input": "str",
  "thought": "str",
  "output": "str",
  "source_ids": ["str"],
  "confidence": 0.0,
  "status": "confirmed"
}
```

### 必須条件
- `status` は `confirmed` のみ
- `thought` は空でないこと
- `confidence` は閾値以上であること
- 長文は事前に分割済みであること

## 学習フロー
1. `JSONL` を読み込む
2. 各行を検証する
3. 学習可能な例だけを抽出する
4. `input / thought / output` を学習フォーマットへ変換する
5. 学習を実行する
6. LoRA アダプタを保存する

## モデル契約
- ベースモデルは学習可能な causal language model とする。
- 学習は LoRA / PEFT を使う。
- trainer は外部依存として注入可能にする。

## エラー時の挙動
- `JSONL` が読めない場合は明示的に失敗する。
- 無効な行はスキップし、学習全体はできるだけ継続する。
- 学習失敗時は中間生成物を保持する。
- 保存失敗時は学習済み状態を破棄しない。

## 境界条件
- 空データセットは安全に終了する。
- 無効な `thought` を含む例は除外する。
- `conflicted` は学習対象にしない。

## 実装上の留意点
- 記憶圧縮モジュールと依存関係を持たせすぎない。
- 学習フォーマットへの変換は単純で追跡可能にする。
- 保存処理と学習処理は分離する。

## 仕様テスト観点
- `JSONL` を読み込めること
- 無効例が除外されること
- 学習後にアダプタが保存されること
- 保存済みアダプタを再ロードできること
- 空データセットで安全に終了すること

## 非目標
- 記憶圧縮
- `DB1` / `DB2` の管理
- 学習データ生成
- 学習品質の最適化

## 補足
- この段階は、既存の JSONL 生成結果を学習可能な形へ接続するための薄い後段として定義する。
