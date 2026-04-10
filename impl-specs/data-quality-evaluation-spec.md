# Data Quality Evaluation 設計メモ

## 目的
`sleep_consolidation` と `sleep_consolidation_training` の間で、学習用データの品質を評価し、誤保存・矛盾・低品質例を学習前に検出する。

## 位置づけ
- 記憶圧縮: `sleep_consolidation.py`
- JSONL 生成: `sleep_consolidation.py`
- `thought` 品質保証: `thought-quality-assurance-spec.md`
- データ品質評価: 本書
- 学習接続: `sleep_consolidation_training.py`

この段階は、学習に回す前のデータセット全体を健全化する監査層として扱う。

## 公開インターフェース

### `DataQualityEvaluator`
- 学習前データセット全体を評価する薄い評価器とする。

### `evaluate(dataset)`
- データセットの健全性を評価し、要約を返す。

### `filter_dataset(dataset)`
- 品質基準を満たす例だけを残す。

### `summarize_issues(dataset)`
- 代表的な問題を集計して返す。

## 評価対象
- `source_ids` の有無と妥当性
- `confidence` の分布
- `status` の偏り
- `thought` の空欄率
- `input / thought / output` の長さバランス
- 重複率
- 矛盾率

## 評価ルール
- `confirmed` 以外が混ざる場合は警告する
- 低信頼例が多すぎる場合は警告する
- 空 `thought` が存在する場合は不合格にする
- 同一内容の重複が多すぎる場合は警告する
- `input` と `output` に対して `thought` の長さが極端に偏る場合は警告する

## スコアの考え方
- 完全性
- 一貫性
- 冗長性
- 追跡可能性
- 再利用可能性

## エラー時の挙動
- 形式不正なデータは除外する
- 評価不能な例は不合格にする
- 例外が起きても、可能な限り集計は続ける

## 境界条件
- 空データセットは安全に終了する
- すべて不合格のデータセットも要約を返す
- 重複がゼロでも問題ない

## 仕様テスト観点
- 空データセットで安全に終了すること
- 空 `thought` が問題として集計されること
- `confirmed` 以外の混入が警告されること
- 重複率が評価されること

## 非目標
- 学習そのもの
- `thought` の意味理解
- 人間監査の完全代替

## 補足
- この段階は、学習に回す前に「どの程度信頼できるデータか」を数えるために用いる。
