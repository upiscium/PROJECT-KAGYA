# ADR 0015: EmbodiedEmotion を疑似身体状態による情動モデュレーション層として採用する

## Status
Accepted

## Context
`impl-specs/embodied-emotion-spec.md` では、`emotion_engine.py` の情動更新に疑似身体状態を加えて、応答の一貫性と文脈性を高める拡張層が定義されている。疲労、負荷、安定性、回復度のような内部状態を持たせることで、同じ `loss` でも状況に応じて反応を変えられるようにしたい。

## Decision
`EmbodiedEmotion` は次の方針で実装する。

- `fatigue` / `load` / `stability` / `recovery` を疑似身体状態として保持する
- 会話、睡眠、失敗、成功などのイベントで身体状態を更新する
- `modulate_emotion(loss, emotion_state)` で身体状態を反映した情動を返す
- 身体状態はクランプし、外部センサーなしでも動作可能にする
- 直接的な人間模倣ではなく、内部状態による反応の一貫性向上を目的とする

## Consequences
### Positive
- 返答の温度感や振れ幅に文脈を持たせやすい
- `loss` 単独よりも応答の説明可能性が上がる
- 疲労や回復といった概念で状態変化を整理しやすい

### Negative
- 人間らしさの評価が曖昧になりやすい
- 状態変数が増えると調整コストが上がる
- 外部センサーがないため、あくまで擬似身体に留まる

## Alternatives Considered
- `EmotionEngineAllostasis` だけで十分とする
  - 単純だが、反応の文脈性を持たせにくい
- 生理信号や外部センサーを統合する
  - より身体性は高いが、スコープと依存が大きすぎる

## Related Tests
- 疲労や負荷で反応が変わる
- 安定性が高いと振れ幅が抑えられる
- 回復イベントで落ち着きが戻る
- 空イベントでも安全に終了する
