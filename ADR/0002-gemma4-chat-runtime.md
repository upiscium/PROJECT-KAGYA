# ADR 0002: Unsloth ベースCLIでの Gemma4 チャット実行環境

## ステータス

承認済み

## 背景

`project-kagya-chat` CLI は、Nix/direnv の開発シェル内で `google/gemma-4-E4B-it` を安定して動かす必要があった。初期構成では、実行時ツールの不足、互換性のないプロンプト形式、モデル出力の漏れが表面化した。

## できること

この作業により、次の機能を持つローカルチャット実行環境が利用できる。

- `uv run project-kagya-chat` から Gemma4 チャットを起動できる
- 開発シェル内で Unsloth を使って instruction-tuned の Gemma4 モデルを読み込める
- 一般的な `User:/Assistant:` 形式ではなく Gemma4 の turn token を使ってプロンプトを整形できる
- 表示前にモデル応答から制御トークンのノイズを除去できる
- Nix と `direnv` で環境を自己完結させられる

## アーキテクチャ

実行環境は3層に分かれる。

1. **環境層**
   - Nix の dev shell が `uv`、CUDA ライブラリ、`openssl`、Triton 互換の CUDA ライブラリパスを提供する。
   - `TRITON_LIBCUDA_PATH` で Triton にホスト側GPUドライバライブラリを指させる。

2. **モデル読込層**
   - `project_kagya.chat` は `unsloth.FastLanguageModel` 経由で `google/gemma-4-E4B-it` を読み込む。
   - チャット経路は Unsloth のみに固定する。

3. **プロンプト・応答層**
   - 利用可能なら Gemma4 の turn marker でプロンプトを構成する。
   - 入力テンソル長を追跡し、新しく生成されたトークンだけをデコードする。
   - 応答の後処理で制御トークンの漏れや空のラッパー出力を取り除く。

## 決定

Gemma4 の instruction-tuned モデルを、Gemma ネイティブのプロンプト形式と Nix ベースのGPU実行環境で使う。

## 理由

- Gemma4 の `-it` 系はチャット用途を想定している。
- 手書きの対話ログよりも、ネイティブな turn token の方が checkpoint に合う。
- CUDA とコンパイラツールの互換性は dev shell 側で管理する必要がある。
- このリポジトリが望む推論経路は Unsloth が提供している。

## 影響

- `google/gemma-4-E4B` のような base checkpoint はチャット用として拒否される。
- dev shell はホストGPUドライバの利用可能性に依存する。
- チャット出力は表示しやすくなるが、依然としてモデル生成でありスキーマ検証済みではない。

## 検証

- `pytest`
- `uv run project-kagya-chat`
