# EmotionEngineAllostasis 設計メモ

## 目的
`loss` を入力として、`Valence` / `Arousal` / `optimal_loss` を更新する情動・アロスタシスモジュールを定義する。

## 公開インターフェース

### `EmotionEngineAllostasis`
- 状態を保持するクラスとする。
- 初期状態は `valence=0.0`, `arousal=0.0`, `optimal_loss=2.5`, `adaptation_rate=0.15` とする。

### `update(loss)`
- `loss` を受け取り、内部状態を更新する。
- 返り値は更新後の状態を表す構造体または同等の値とする。

### `get_state()`
- 現在の `valence`, `arousal`, `optimal_loss` を返す。

## 更新ルール

### Arousal
`A_new = max(0.0, min(1.0, A_current * 0.8 + loss * 0.2))`

### Wundt curve
`W = 1.0 - 0.3 * (loss - L_opt)^2`

### Valence
`V_new = max(-1.0, min(1.0, V_current * 0.4 + W * 0.6))`

### optimal_loss
`L_opt_new = (1.0 - alpha) * L_opt_current + alpha * loss`

## 境界条件
- `loss` が極端値でも、`valence` は `[-1.0, 1.0]`、`arousal` は `[0.0, 1.0]` に収める。
- `optimal_loss` は実数として更新し、クランプは行わない。
- `loss` が負値でも例外にせず、数式どおりに処理する。

## エラー時の挙動
- 入力が数値でない場合は例外を返すか、既存パターンに合わせて明示的に失敗する。
- 状態更新は副作用を持つため、失敗時は状態を更新しない。

## 仕様テスト観点
- 初期値が仕様どおりであること
- `loss` から `valence` / `arousal` / `optimal_loss` が数式どおり更新されること
- `valence` が `[-1.0, 1.0]` にクランプされること
- `arousal` が `[0.0, 1.0]` にクランプされること
- `loss` が極端値でも状態更新が破綻しないこと

## 非目標
- 完全な感情モデルの再現
- 生理学的妥当性の厳密検証
- 外部状態や記憶システムとの統合ロジック

## 補足
- これは「感情表現」よりも「会話制御用の状態量」として扱う。
