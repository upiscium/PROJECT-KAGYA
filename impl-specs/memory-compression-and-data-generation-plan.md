# Memory Compression and Data Generation 実装計画

## 目的
`sleep_consolidation.py` 周辺で、`DB1` の会話履歴を圧縮し、学習用データを生成する流れを段階的に実装する。

## ゴール
- `DB1` の生ログから圧縮済みの事実候補を抽出できる
- `confirmed` のみを長期保存・学習候補にできる
- 学習用 `JSONL` データを生成できる
- 学習処理は後段に分離し、まずはデータ生成までを安定化する

## 非目標
- 完全な自律学習
- `thought` の正解性の保証
- 長期記憶の全件再構成
- 学習ループの自動反復

## 実装方針
- 圧縮とデータ生成を分離する
- 原文を消さずに、参照 ID と信頼度を保持する
- `confirmed` / `tentative` / `conflicted` を明示的に扱う
- 自己生成データの循環を避けるため、生成前後に再フィルタを入れる

## 公開インターフェース案

### `SleepCycleManager`
- 睡眠フェーズ全体を管理する薄いオーケストレータ

### `triage_episodes(...)`
- `DB1` から情動の大きいエピソードを抽出する

### `extract_semantic_candidates(...)`
- 抽出済みエピソードを事実候補に変換する

### `generate_dream_dataset(...)`
- 学習用の `input / thought / output` を `JSONL` 化する

### `save_adapter(...)`
- 学習完了後の LoRA アダプタ保存を担う

## データフロー
1. `DB1` から候補を集める
2. LLM で事実候補を抽出する
3. `confidence` と `status` でフィルタする
4. 学習候補の `input / thought / output` を作る
5. 長すぎる項目を分割する
6. `JSONL` として保存する
7. 必要に応じて後段の学習に渡す

## データスキーマ
```json
{
  "input": "str",
  "thought": "str",
  "output": "str",
  "source_ids": ["str"],
  "confidence": 0.0,
  "status": "confirmed"
}
```

## 段階的実装

### Phase 1: 圧縮基盤
- `DB1` から候補を選ぶ
- `source_ids` / `confidence` / `status` を持つ記録形式を扱う
- `confirmed` のみを残す

### Phase 2: データ生成
- 圧縮済み記憶から `input / thought / output` を生成する
- 再フィルタして JSONL に落とす
- 長文分割を導入する

### Phase 3: 学習接続
- JSONL を学習器へ渡す
- 学習と保存は分離する
- 失敗しても圧縮済みデータを保持する

## 境界条件
- 空の `DB1` は失敗ではなく空結果として扱う
- `thought` が空または不正な場合は除外する
- `confidence` が低い候補は学習対象にしない
- `conflicted` は保存・学習のどちらにも回さない

## エラー時の挙動
- 抽出失敗時は元データを保持する
- JSONL 出力失敗時は中間データを破棄しない
- 学習失敗時も圧縮済みデータを再利用できるようにする

## 仕様テスト観点
- `confirmed` のみが長期保存対象になること
- `confidence` が閾値未満の候補が除外されること
- JSONL の各行がスキーマを満たすこと
- 空の `DB1` で安全に終了すること
- 長文が分割されること

## 実装順序
1. 圧縮用の記録形式を整える
2. `DB1` からの triage を実装する
3. 事実候補抽出を実装する
4. JSONL 生成を実装する
5. 学習接続を最後に追加する

## まとめ
- まずは「圧縮の精度」を固める
- その後に「データ生成」を安定させる
- 学習は最後に接続する
