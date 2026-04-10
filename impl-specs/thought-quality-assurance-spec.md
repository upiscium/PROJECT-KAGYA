# Thought Quality Assurance 設計メモ

## 目的
`sleep_consolidation` の学習データに含まれる `thought` の品質を検査し、誤り・空欄・不整合を学習前に除外する。

## 位置づけ
- 記憶圧縮: `sleep_consolidation.py`
- JSONL 生成: `sleep_consolidation.py`
- 学習接続: `sleep_consolidation_training.py`
- `thought` 品質保証: 本書

この段階は、学習データの中でも特に壊れやすい `thought` を検査するゲートとして扱う。

## 公開インターフェース

### `ThoughtQualityAssurer`
- `thought` を検査して学習可否を判定する薄い検査器とする。

### `validate_example(example)`
- `input / thought / output` を持つ例を検証する。
- 返り値は検証結果の要約とする。

### `filter_examples(examples)`
- 学習に使える例だけを残す。
- 不正例は除外する。

### `score_thought(thought)`
- `thought` の品質を簡易スコアとして返す。

## 品質条件
- `thought` が空でないこと
- `thought` が入力や出力と矛盾しないこと
- `thought` が過度に長すぎないこと
- `thought` が再生産不能な雑音だけで構成されていないこと
- `thought` が `confirmed` 由来の文脈と整合すること

## 検査ルール
- 空文字列は除外する
- 長すぎる `thought` は除外または分割対象とする
- 低スコアの `thought` は学習対象にしない
- `input` / `output` と明らかに矛盾する `thought` は除外する

## スコアの考え方
- 長さ
- 空白率
- 繰り返し率
- 入出力との整合性
- 参照元の信頼度

## エラー時の挙動
- 型が不正な場合は明示的に失敗する
- 検査不能な例は除外する
- 例外が起きても、他の例の検査は継続する

## 境界条件
- 空の `examples` は安全に終了する
- `thought` が短いが有効な場合は通す
- `thought` が長い場合はルールに従って扱う

## 仕様テスト観点
- 空の `thought` が除外されること
- 長すぎる `thought` が除外されること
- `input` / `output` と矛盾する例が除外されること
- 空の入力リストで安全に終了すること

## 非目標
- 完全な意味理解
- 人間による採点の再現
- モデル生成品質そのものの改善

## 補足
- ここでの目的は、学習前に壊れた `thought` を落とすことに限定する。
