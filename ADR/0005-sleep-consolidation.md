# ADR 0005: SleepCycleManager を圧縮と JSONL 生成に限定した睡眠フェーズ管理器として採用する

## Status
Accepted

## Context
`impl-specs/memory-compression-and-data-generation-plan.md` では、`sleep_consolidation.py` 周辺で `DB1` の会話履歴を圧縮し、学習用データを生成する流れが定義されている。ただし、MVP では学習接続まで一気に入れず、まずは圧縮基盤と JSONL 生成までを安定化する方針が採られている。

このため、`SleepCycleManager` は `DB1` 相当のエピソード一覧を入力として受け取り、情動が大きい候補を triage し、`confirmed` のみを圧縮・学習候補として JSONL に書き出す薄いオーケストレータとして扱う。

## Decision
`SleepCycleManager` は次の方針で実装する。

- 入力は `DB1` 相当のエピソード一覧とする
- `triage_episodes(...)` で情動の大きいエピソードを抽出する
- `extract_semantic_candidates(...)` で `confirmed` の事実候補のみを残す
- `generate_dream_dataset(...)` で `input / thought / output` の JSONL を生成する
- 長すぎる例は分割し、低信頼・矛盾候補は除外する
- 学習接続やアダプタ保存は、後段の責務として分離する

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
