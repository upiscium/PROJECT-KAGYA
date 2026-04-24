# ADR 0001: Gemma 4 E4B の微調整に Unsloth の 4bit QLoRA を使う

## ステータス

承認済み

## 背景

`google/gemma-4-E4B` を限られたVRAMで動かせる、実用的な微調整ワークフローが必要だった。
このリポジトリはすでに `transformers`、`peft`、`bitsandbytes` に依存しているため、主な判断点は学習スタックの組み方だった。

## 決定

Unsloth の 4bit ロードを既定パスとして使い、`google/gemma-4-E4B` の上に QLoRA アダプタを学習するCLIを提供する。

## 理由

- Unsloth は低VRAMでの微調整を簡単にし、定型コードを減らせる。
- 4bit ロードはこのクラスのモデルで最もメモリ効率のよい既定値である。
- `trl.SFTTrainer` はテキストベースの教師あり微調整にそのまま適合する。
- 対象モジュールの一覧は標準的な Gemma の projection 層に一致し、アダプタの範囲を絞れる。

## 影響

- `bitsandbytes` と Unsloth に対応したGPUスタックが必要になる。
- 初期実装は `text` カラムを前提とし、chat template の前処理は含まない。
- 学習設定はCLI駆動なので、LoRA rank、sequence length、batch size、learning rate をコード編集なしで調整できる。

## 検討した代替案

- 純粋な `transformers` + `peft`
  - 同じ QLoRA フローでも手作業のセットアップが増えるため不採用。
- フル精度での微調整
  - 大規模モデルを限られたハードウェアで試す用途に向かないため不採用。
