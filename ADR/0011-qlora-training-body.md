# ADR 0011: QLoRATrainer を JSONL 後段の学習実行器として採用する

## Status
Accepted

## Context
`impl-specs/qlora-training-body-spec.md` では、`sleep_consolidation` が生成した JSONL データを用いて学習を実行し、LoRA/QLoRA アダプタを保存・再読込する後段の学習実行器が定義されている。4bit モデルのロードは別段階として扱い、ここでは学習ループ、前処理、保存・再読込を担う。

## Decision
`QLoRATrainer` は次の方針で実装する。

- `input / thought / output` を `<think>` フォーマットの prompt に変換する
- `confirmed` かつ高信頼で、空でない `thought` を持つ例のみ学習対象にする
- 学習前に `peft.prepare_model_for_kbit_training` 相当の前処理を適用できる
- 学習実行は注入された backend に委譲できる
- 学習後にアダプタを保存し、再読込できる
- 4bit モデルの読み込み自体は別段階とする

## Consequences
### Positive
- 学習実行・保存・再読込の責務が明確になる
- JSONL 後段の学習を薄い実行器として扱える
- 4bit ロードと学習ループを分離できる

### Negative
- 4bit モデルのロードは別実装が必要になる
- 実学習の品質は backend に依存する
- `thought` の品質が悪いと学習効果が落ちる

## Alternatives Considered
- 4bit ロードまで本 ADR に含める
  - 実装範囲が広がりすぎるため分離した
- 学習を固定実装だけで閉じる
  - テストはしやすいが、backend 差し替えが難しくなる

## Related Tests
- prompt が `<think>` 形式になる
- 空 `thought` の例が除外される
- 学習後にアダプタが保存される
- 保存済みアダプタを再読込できる
