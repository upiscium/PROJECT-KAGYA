# ADR 0014: MultimodalIngestAPI を画像・音声・動画・テキストの統一入口として採用する

## Status
Accepted

## Context
`impl-specs/multimodal-fastapi-interface-spec.md` では、画像、音声、動画、テキストを受け取る FastAPI ベースの統一入力インタフェースが定義されている。後段の推論、記憶、学習に渡す前の入口として、モダリティごとの責務を分離しつつ、ひとつの API として扱える必要がある。

## Decision
`MultimodalIngestAPI` は次の方針で実装する。

- `POST /ingest` でテキストとファイルをまとめて受ける
- `POST /chat` でテキスト中心の対話入力を受ける
- `WS /stream` で逐次応答を扱う
- 画像、音声、動画は `UploadFile` として受ける
- 受信後の重い処理は API 本体から外し、入力正規化層を挟む
- FastAPI を本依存として追加する
- `python-multipart` を依存に加えて form/file 入力を成立させる

## Consequences
### Positive
- 入力経路を統一できる
- テキスト、ファイル、ストリーミングの責務を分けられる
- 後段の推論・記憶・学習へつなぎやすい

### Negative
- モダリティごとの制限や検証が増える
- 動画や音声の大きなファイルは非同期化が必要になる
- `fastapi` と `python-multipart` への依存が増える

## Alternatives Considered
- 単一のエンドポイントに全モダリティを詰め込む
  - 実装は短いが、責務が混ざる
- モダリティごとに完全に別 API にする
  - 明確だが、入口としての一体感が弱い

## Related Tests
- テキストのみの受理ができる
- 画像、音声、動画の受理ができる
- 空入力が失敗する
- サイズ超過が拒否される
- WebSocket で逐次応答できる
