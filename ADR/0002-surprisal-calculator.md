# ADR 0002: SurprisalCalculator を推論専用のマスク付き loss 計算器として採用する

## Status
Accepted

## Context
`impl-specs/surprisal-calculator-spec.md` では、`context_text` と `new_input` からベース LLM の loss を使って入力の驚き度を計算する `SurprisalCalculator` が定義されている。目的は学習ではなく、情動エンジンへ渡す制御信号を得ることにある。

このモジュールは、過去の文脈を学習対象に含めないために `labels` で `-100` マスクを行い、推論専用として `torch.no_grad()` を前提にする。

## Decision
`SurprisalCalculator` は次の方針で実装する。

- `context_text` と `new_input` を受け取り、モデル入力用の構造を組み立てる
- `context_text` に対応する token はすべて `-100` でマスクする
- `new_input` の token のみを loss 計算対象とする
- `compute_surprisal(loss_inputs)` は forward pass の loss を単一の数値として返す
- 勾配計算は行わず、推論専用の軽量モジュールとして扱う
- 空の `context_text` には対応するが、空の `new_input` は失敗させる

## Consequences
### Positive
- 自己発話や過去文脈を誤って学習対象に入れにくい
- 情動エンジンへの制御信号が明確になる
- 推論専用なので、実行コストと責務が小さい

### Negative
- tokenization 境界のずれに弱い
- loss は文長やテンプレートの影響を受けるため、値の単純比較は難しい
- 毎ターン計算すると重くなる可能性がある

## Alternatives Considered
- context をマスクせず全体で loss を取る
  - 実装は簡単だが、自己発話や過去文脈の影響を受けやすい
- 驚き度をルールベースで近似する
  - 軽いが、ベース LLM の予測誤差という仕様から外れる

## Related Tests
- `context_text` が `-100` で正しくマスクされる
- `new_input` が loss 計算対象に含まれる
- 空の `context_text` で動作する
- 非数値入力が失敗する
- `torch.no_grad()` 前提で動く
