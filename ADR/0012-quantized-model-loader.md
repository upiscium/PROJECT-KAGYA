# ADR 0012: QuantizedModelLoader を 4bit ロードの責務分離層として採用する

## Status
Accepted

## Context
`impl-specs/4bit-quantized-model-load-spec.md` では、QLoRA 学習の前提として、ベースモデルを 4bit 量子化で安定して読み込む責務が定義されている。`sleep_consolidation_training` や `qlora_training` からは独立した、モデルロード専用の薄い層が必要である。

## Decision
`QuantizedModelLoader` は次の方針で実装する。

- 量子化設定を `QuantizationConfig` として明示する
- `load_4bit_model(...)` で 4bit ロードを行う
- `prepare_for_training(...)` で学習前処理を適用する
- `bitsandbytes` / CUDA の失敗は明示的に扱う
- 4bit ロードができない場合に不完全なモデルを返さない

## Consequences
### Positive
- 4bit ロードを学習本体から分離できる
- 設定が明示され、責務が追いやすい
- QLoRA 学習の前提を単独で検証しやすい

### Negative
- GPU / CUDA / bitsandbytes 依存が強い
- テスト環境での再現性が低い
- 学習とは別にロード失敗のハンドリングが必要になる

## Alternatives Considered
- 4bit ロードを `QLoRATrainer` に埋め込む
  - 実装は短いが、責務が混ざる
- ロード処理を省略し、常に通常モデルを使う
  - 実運用の VRAM 節約にならない

## Related Tests
- 量子化設定が構築される
- 空のモデル名で失敗する
- 学習前処理と接続できる
- ロード失敗時に明示的に失敗する
