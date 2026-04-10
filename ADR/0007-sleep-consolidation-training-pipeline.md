# ADR 0007: SleepConsolidationTrainingPipeline を JSONL 後段の学習接続器として採用する

## Status
Accepted

## Context
`impl-specs/sleep-consolidation-training-design.md` では、`sleep_consolidation.py` が生成した JSONL を使って学習を実行し、LoRA アダプタを保存・再読込する後段パイプラインが定義されている。

この段階は、記憶圧縮や JSONL 生成とは切り離し、学習可能な例だけを読み込んで学習と保存を行う薄い接続層として扱う。

## Decision
`SleepConsolidationTrainingPipeline` は次の方針で実装する。

- 入力は JSONL データセットのパスとする
- 各行を検証し、`confirmed` かつ高信頼で、空でない `thought` を持つ例のみ学習対象にする
- 学習は外部注入された trainer に委譲する
- 学習後は LoRA アダプタを保存し、必要に応じて再読込できる
- 無効行はスキップし、空データセットは安全に終了する

## Consequences
### Positive
- 学習接続の責務が明確になる
- 圧縮・生成・学習を段階的に検証できる
- 無効データの混入を抑えやすい

### Negative
- 学習品質の改善は trainer 側の責務に依存する
- JSONL のスキーマが崩れると学習対象が減る
- 記憶圧縮とは別にデータ品質管理が必要になる

## Alternatives Considered
- 学習処理を `SleepCycleManager` に統合する
  - 実装は短いが、責務が肥大化する
- JSONL を読み込まず、メモリ上のオブジェクトだけで学習する
  - 実験はしやすいが、後段の再現性が弱い

## Related Tests
- JSONL を読み込める
- 無効例が除外される
- 学習後にアダプタが保存される
- 保存済みアダプタを再ロードできる
- 空データセットで安全に終了する
