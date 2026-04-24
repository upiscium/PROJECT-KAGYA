# ADR 0003: マルチモーダル添付を HTTP と WebSocket で受ける

## ステータス

承認済み

## 背景

`project-kagya-chat` 由来のマルチモーダル入力を HTTPS API 側へ持ち込みたい一方で、利用者はファイルパスではなく実ファイルをそのまま送信したい場面がある。
また、WebSocket 経由でも同じ入力セマンティクスを維持しつつ、低遅延なやり取りを扱えるようにしたい。

## 決定

HTTP では `multipart/form-data` によるファイルアップロードを受け付け、WebSocket では JSON 内の inline 添付 `filename + data(base64)` を受け付ける。
既存の `text + attachments[]` 形式も互換用として残す。

## 理由

- 実ファイルを送れるので、サーバー側のパス依存を減らせる。
- HTTP と WebSocket のどちらでも同じマルチモーダル意味論を維持できる。
- 既存の JSON パス指定を残せば、後方互換を壊しにくい。
- 添付を一時ファイルに落としてから処理すれば、既存のモデル前処理を大きく変えずに済む。

## 影響

- サーバー側で一時ファイル管理が必要になる。
- WebSocket の添付は base64 になるため、JSON 本文はやや大きくなる。
- 以前のパス指定クライアントも引き続き利用できる。

## 検証

- `uv run pytest tests/test_api.py -v`
- `uv run ruff check src/project_kagya/api.py tests/test_api.py`
