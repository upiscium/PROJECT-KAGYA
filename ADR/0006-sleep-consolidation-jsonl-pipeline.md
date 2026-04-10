# ADR 0006: SleepCycleManager を圧縮と JSONL 生成に限定し、学習接続を後段へ分離する

## Status
Accepted

## Context
`impl-specs/memory-compression-and-data-generation-plan.md` では、`sleep_consolidation.py` 周辺で `DB1` の会話履歴を圧縮し、学習用データを生成する流れが定義されている。一方で、学習接続 (`train` / `load_adapter` / `save_adapter`) は別段階に分離し、まずは圧縮精度と JSONL 生成の安定化を優先する方針が示されている。

現在の実装では、`SleepCycleManager` は `DB1` 相当のエピソード一覧を入力として受け取り、情動の大きい候補を triage し、`confirmed` のみを残して `input / thought / output` の JSONL を生成する役割に限定する。

## Decision
`SleepCycleManager` は次の方針で実装する。

- 入力は `DB1` 相当のエピソード一覧とする
- `triage_episodes(...)` で情動の大きいエピソードを抽出する
- `extract_semantic_candidates(...)` で `confirmed` の事実候補のみを残す
- `generate_dream_dataset(...)` で学習用 JSONL を生成する
- 長すぎる例は分割し、低信頼・矛盾候補は除外する
- `train` / `load_adapter` / `save_adapter` の学習接続は後段の責務として残す

## Consequences
### Positive
- 圧縮とデータ生成の責務が明確になる
- 学習処理を分離できるため、失敗の影響範囲を小さくできる
- 自己生成ループに入る前に、データ品質を検査しやすい

### Negative
- 学習まで一気通貫にしないため、完成形にはまだ届かない
- `thought` の品質保証は依然として難しい
- JSONL 生成時のフィルタ設計が重要になる

## Alternatives Considered
- すぐに学習接続まで含める
  - 早く試せるが、圧縮と学習の問題が混ざりやすい
- `DualMemorySystem` を直接注入して記憶層から読む
  - 統合度は高いが、薄い圧縮モジュールとしての責務がぼやける

## Related Tests
- 情動の大きいエピソードだけが triage される
- `confirmed` かつ高信頼の候補のみが残る
- JSONL がスキーマどおりに書き出される
- 空入力でも安全に終了する
