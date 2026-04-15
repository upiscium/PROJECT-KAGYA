# ADR 0013: MultimodalIngestAPI を統一入力の入口として採用する

## Status
Accepted

## Context
`impl-specs/multimodal-fastapi-interface-spec.md` では、画像・音声・動画・テキストを受け取る FastAPI ベースの統一入力インタフェースが定義されている。現状のシステムでは、モダリティごとの入力を整理し、後段の推論・記憶・学習へつなぐための入口が必要である。

## Decision
`MultimodalIngestAPI` は次の方針で扱う。

- `POST /ingest` で画像・音声・動画・テキストをまとめて受ける
- `POST /chat` でテキスト中心の対話入力を受ける
- `WS /stream` で逐次応答を扱う
- モダリティごとに `UploadFile` とテキストを分離して受ける
- 受信後の重い処理は API 本体から外し、入力正規化層を挟む

## Consequences
### Positive
- 入力経路が統一される
- テキスト、ファイル、ストリーミングの責務が分かれる
- 後段の推論・記憶・学習へつなぎやすい

### Negative
- 動画や音声の大きなファイルは別途非同期処理が必要になる
- モダリティごとの制限や検証が増える
- 単一エンドポイントに全機能を詰めるより実装がやや複雑になる

## Alternatives Considered
- 単一のエンドポイントにすべてのモダリティを詰める
  - 実装は簡単だが、責務が混ざる
- モダリティごとに完全に別 API にする
  - 明確だが、入口としての一体感が弱い

## Related Tests
- テキストのみの受理ができる
- 画像、音声、動画の受理ができる
- 空入力が失敗する
- サイズ超過が拒否される
- WebSocket で逐次応答できる
