# ADR 0009: DataQualityEvaluator を学習前のデータ品質監査層として採用する

## Status
Accepted

## Context
`impl-specs/data-quality-evaluation-spec.md` では、`sleep_consolidation` と `sleep_consolidation_training` の間に、学習前データセット全体を評価し、誤保存・矛盾・低品質例を検出する監査層が定義されている。`thought` の品質保証だけでは、データセット全体の健全性までは十分に担保できない。

## Decision
`DataQualityEvaluator` は次の方針で実装する。

- `evaluate(...)` でデータセット全体の健全性を集計する
- `filter_dataset(...)` で学習に使える例だけを残す
- `summarize_issues(...)` で代表的な問題を数える
- 空 `thought`、低信頼、重複、`conflicted` を検出する
- 学習前の監査層として、除外と警告を優先する

## Consequences
### Positive
- 学習前にデータセット全体の品質を確認できる
- 自己生成データの汚染を減らしやすい
- `thought` 品質保証だけでは見落とす全体傾向を把握しやすい

### Negative
- 評価基準がヒューリスティックに依存する
- 完全な意味理解はできない
- 学習前の前処理段数が増える

## Alternatives Considered
- `thought` 品質保証だけで済ませる
  - 単純だが、データセット全体の偏りを見逃しやすい
- 人間レビューに全面依存する
  - 高品質だが、継続運用が重い

## Related Tests
- 空データセットで安全に終了する
- 空 `thought` が問題として集計される
- `confirmed` 以外の混入が警告される
- 重複率が評価される
