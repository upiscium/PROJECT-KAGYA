# ADR 0008: ThoughtQualityAssurer を学習前の `thought` 品質ゲートとして採用する

## Status
Accepted

## Context
`impl-specs/thought-quality-assurance-spec.md` では、`sleep_consolidation` が生成した学習データに含まれる `thought` を、学習前に検査して除外する品質ゲートが定義されている。`thought` は内部推論の教師として使うため、空欄・長すぎるもの・入力や出力と明らかに矛盾するものを学習に回さない必要がある。

## Decision
`ThoughtQualityAssurer` は次の方針で実装する。

- `ThoughtExample` を検査対象とする
- `validate_example(...)` で空欄、長すぎるもの、不整合を検出する
- `filter_examples(...)` で学習可能な例だけを残す
- `score_thought(...)` で簡易スコアを返す
- 例外を投げるより、理由付きで除外することを優先する

## Consequences
### Positive
- 壊れた `thought` を学習に入れにくくなる
- 学習前にデータ品質を機械的に確認できる
- 低品質な自己生成ループを緩和しやすい

### Negative
- 完全な意味理解はできない
- スコアが過度に単純だと、良い例を取りこぼす可能性がある
- `thought` の品質保証は依然としてヒューリスティックに依存する

## Alternatives Considered
- `thought` を検査せずそのまま学習する
  - 実装は簡単だが、誤りや雑音を固定化しやすい
- 人間レビューのみで品質を判断する
  - 精度は高いが、継続運用が重い

## Related Tests
- 空の `thought` が除外される
- 長すぎる `thought` が除外される
- `input` / `output` と矛盾する例が除外される
- 空の入力リストで安全に終了する
