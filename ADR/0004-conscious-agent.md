# ADR 0004: ConsciousAgent を情動と記憶を橋渡しする薄い生成ラッパーとして採用する

## Status
Accepted

## Context
`impl-specs/conscious-agent-spec.md` では、`ConsciousAgent` が情動状態と検索済み記憶を統合し、推論モデルに渡すシステムプロンプトを構築して応答を生成する役割として定義されている。

このモジュールは、情動値と `DB1` / `DB2` の検索結果を prompt にまとめ、生成モデルへそのまま渡せる形に整える薄いラッパーとして扱う。

## Decision
`ConsciousAgent` は次の方針で実装する。

- `valence`、`arousal`、`memory_context` を含む prompt を構築する
- `memory_context` が空でも動作する
- `<think>` などの内部思考を促す指示は控えめに含める
- `generate_response(prompt)` は生成モデルを呼び出し、生成結果を薄い結果型で返す
- 生成モデルや入力型が不正な場合は明示的に失敗する

## Consequences
### Positive
- prompt 組み立てと生成実行の責務が明確になる
- 情動と記憶を推論モデルに橋渡ししやすい
- 生成結果の扱いを薄い型にまとめられる

### Negative
- `think` 系の挙動はモデル依存で安定しない可能性がある
- prompt が長くなりすぎると応答品質に影響する可能性がある
- 記憶整合性を毎回厳密に詰めると遅くなる

## Alternatives Considered
- 直接モデル呼び出しに情動と記憶を混ぜる
  - 実装は単純だが、責務が分散しやすい
- 内部思考を厳密に強制する
  - 挙動は強くなる可能性があるが、壊れやすい

## Related Tests
- prompt に `valence`、`arousal`、`memory_context` が含まれる
- 空の `memory_context` でも動作する
- 生成失敗時に例外を返す
- 出力フォーマットが安定している
