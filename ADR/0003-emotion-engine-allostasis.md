# ADR 0003: EmotionEngineAllostasis を loss 駆動の状態更新器として採用する

## Status
Accepted

## Context
`impl-specs/emotion-engine-spec.md` では、`loss` を入力として `valence`、`arousal`、`optimal_loss` を更新する `EmotionEngineAllostasis` が定義されている。目的は感情モデルの厳密な再現ではなく、会話制御用の状態量として振る舞う軽量なアロスタシス機構を持つことにある。

このモジュールは、固定された更新式により状態変化を予測可能にし、情動エンジンとしての検証を容易にする。

## Decision
`EmotionEngineAllostasis` は次の方針で実装する。

- 内部状態として `valence`, `arousal`, `optimal_loss` を保持する
- 初期値は `valence=0.0`, `arousal=0.0`, `optimal_loss=2.5`, `adaptation_rate=0.15` とする
- `update(loss)` で以下の式に従って状態を更新する
  - `A_new = max(0.0, min(1.0, A_current * 0.8 + loss * 0.2))`
  - `W = 1.0 - 0.3 * (loss - L_opt)^2`
  - `V_new = max(-1.0, min(1.0, V_current * 0.4 + W * 0.6))`
  - `L_opt_new = (1.0 - alpha) * L_opt_current + alpha * loss`
- `loss` が数値でない場合は明示的に失敗する
- `update` は更新後の状態を返す

## Consequences
### Positive
- 更新挙動が固定され、テストしやすい
- `loss` を情動制御にそのまま接続しやすい
- `valence` と `arousal` の境界が明確になる

### Negative
- 感情モデルとしては単純であり、意味論は薄い
- `loss` の振れに対して反応が直線的になりやすい
- `optimal_loss` の更新が速すぎると状態が不安定になる

## Alternatives Considered
- より複雑な感情状態モデルを採用する
  - 表現力は上がるが、仕様と検証が難しくなる
- `loss` を直接応答のトーンに変換する
  - 単純だが、状態保持による一貫性が得にくい

## Related Tests
- 初期値が仕様どおりである
- `loss` に応じて `valence` / `arousal` / `optimal_loss` が更新される
- `valence` が `[-1.0, 1.0]` にクランプされる
- `arousal` が `[0.0, 1.0]` にクランプされる
- 非数値入力が失敗する
