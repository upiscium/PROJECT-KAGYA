# SurprisalCalculator 設計メモ

## 目的
`context_text` と `new_input` から、ベース LLM の loss を用いて入力の驚き度を計算する。

## 公開インターフェース

### `SurprisalCalculator`
- `loss` を会話制御用の信号として扱う軽量モジュールとする。
- 勾配計算は行わない。

### `build_inputs(context_text, new_input)`
- `context_text` と `new_input` を受け取り、トークン化前の入力構造を作る。
- 返り値は、モデル入力用テキストとラベルマスク情報を含む構造体とする。

### `compute_surprisal(loss_inputs)`
- モデルの forward pass から loss を計算する。
- 返り値は単一の数値とする。

## 入力ルール
- `context_text` は過去の会話文脈。
- `new_input` は今回評価対象の新規入力。
- `context_text` 部分は学習対象に含めず、`labels` で `-100` マスクする。

## マスクルール
- `context_text` に対応する token はすべて `-100` にする。
- `new_input` に対応する token は通常ラベルを使う。
- 境界は tokenization 後の位置で決定する。

## 境界条件
- 空の `context_text` でも動作する。
- 空の `new_input` は明示的に失敗するか、既存パターンに合わせて扱う。
- tokenization のずれでマスク位置が崩れないよう、入力境界を固定する。

## エラー時の挙動
- モデル入力の構築に失敗した場合は例外を返す。
- `loss` 計算に失敗した場合は状態を変えずに失敗する。
- 数値以外の入力は受け付けない。

## 仕様テスト観点
- `context_text` が `-100` で正しくマスクされること
- `new_input` が loss 計算対象に含まれること
- 空の `context_text` で動作すること
- 非数値入力が失敗すること
- 推論専用で `torch.no_grad()` を前提にできること

## 非目標
- 学習そのもの
- モデルの再学習
- 驚き度の心理学的正確性

## 補足
- このモジュールの役割は、情動エンジンへ渡す制御信号の算出に限定する。
