# ADR 0010: DriftController を継続学習の安全弁として採用する

## Status
Accepted

## Context
`impl-specs/drift-control-spec.md` では、`sleep_consolidation_training` による継続学習でモデルの挙動が崩れるのを抑制するため、学習前後の状態を監視し、必要なら更新を拒否またはロールバックする `DriftController` が定義されている。

継続学習は有用だが、繰り返すほど出力トーンや語彙、応答長の分布が崩れる可能性があるため、最後の安全弁が必要である。

## Decision
`DriftController` は次の方針で実装する。

- 学習前の状態を snapshot として保存する
- 学習前後の差分を `DriftReport` として測定する
- 閾値を超えた場合は更新を拒否する
- 必要に応じて snapshot へ rollback する
- 学習回数が多い場合は判定をより厳しくする

## Consequences
### Positive
- 継続学習の暴走を抑えやすい
- 学習前後の比較が明示化される
- 更新の受理・拒否を機械的に判定しやすい

### Negative
- ドリフト検知はヒューリスティックに依存する
- 完全な性能保証はできない
- 監視とロールバックの運用コストが増える

## Alternatives Considered
- ドリフト対策を行わず継続学習する
  - 実装は簡単だが、劣化の抑制ができない
- 人間監査だけで更新を許可する
  - 精度は高いが、継続運用が重い

## Related Tests
- snapshot と rollback が機能する
- 閾値超過時に更新が拒否される
- 低ドリフト時に更新が受け入れられる
- 初回学習でベースラインが確立される
