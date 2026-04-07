# PROJECT-KAGYA 実装計画

## 目的
- 仕様書 `impl-specs/v1.0.md` に沿って、主観AIアーキテクチャの最小実装から統合実装までを段階的に進める。
- 各フェーズは単体で検証可能な最小成果物を作り、最後に統合チャットループと睡眠学習まで接続する。
- 現状はコア実装、設定駆動 CLI、settings.toml、主要テストまで完了している。

## 実装方針
- まずはダミー入力とモックを使って、数式・入出力・データフローを固定する。
- 重いモデル依存は後回しにし、API形状とテストを先に固める。
- 仕様にあるクラス名・メソッド名・保存形式を優先し、命名を崩さない。
- 各フェーズ完了時に `just lint` / `just typecheck` / `just test` のうち必要なものを実行する。
- 以後の設定は `settings.toml` を単一ソースとして扱う。

## 現状サマリ
- `emotion_engine.py` 実装済み
- `surprisal_calculator.py` 実装済み
- `dual_memory_system.py` 実装済み
- `conscious_agent.py` 実装済み
- `main.py` 実装済み
- `sleep_consolidation.py` 実装済み
- `settings.py` / `cli.py` / `settings.toml` 実装済み
- `just check-all` と settings 起動テストは通過済み

## フェーズ0: 基盤整備
### タスク
- `src/project_kagya/` を作成する。
- `tests/` を作成する。
- パッケージ初期化ファイルと共通型定義を整える。
- 仕様で必要な依存が `pyproject.toml` に入っているか確認する。
- 既存の CLI エントリポイントがあるなら、将来の統合先として残す。
- `settings.toml` を追加し、runtime / model / memory / emotion / sleep / logging / paths を管理する。

### 完了条件
- プロジェクト内のモジュールを import できる。
- テスト配置先ができている。
- 設定ファイルから runtime を起動できる。

## フェーズ1: 予測誤差と情動のコアロジック
### 1-1 surprisal_calculator.py
#### タスク
- LLM の tokenizer / model を受け取る計算関数を定義する。
- `context_text` 部分を `labels = -100` でマスクするロジックを実装する。
- `torch.no_grad()` 内で loss を計算する。
- 入力テキストから context と新規入力を安全に結合するヘルパーを作る。
- モデル依存を隠すため、テストではダミーモデルで loss 計算を検証する。

#### 状況
- 実装済み。

### 1-2 emotion_engine.py
#### タスク
- `EmotionEngineAllostasis` クラスを実装する。
- 状態変数 `optimal_loss` と `adaptation_rate` を保持する。
- Arousal 更新式をそのまま実装する。
- Wundt 曲線の valence 変換を実装する。
- Valence 更新式をそのまま実装する。
- `optimal_loss` の移動平均更新を実装する。
- 値域を `[-1, 1]` / `[0, 1]` にクランプする。

#### 状況
- 実装済み。

## フェーズ2: Dual Memory の実装
### 2-1 dual_memory_system.py
#### タスク
- ChromaDB クライアント初期化を実装する。
- `hippocampus` と `cortex` の2コレクションを管理する。
- episodic 保存用のメタデータスキーマを定義する。
- `save_episodic(user_input, response, valence, arousal)` を実装する。
- `retrieve_context(query)` で DB1 / DB2 を別々に検索し、統合文字列として返す。
- `consolidate_to_semantic(llm_pipeline)` で DB1 の記録を要約・事実抽出し、DB2 に移行する。
- 移行後に DB1 側レコードを削除する。

#### 状況
- 実装済み。

### 2-2 記憶データ整形
#### タスク
- 検索結果をプロンプト挿入しやすいフォーマットに統一する。
- 情動メタデータを表示用文字列へ整形する。
- 事実記憶とエピソード記憶の区別を明確にする。

#### 状況
- 実装済み。

## フェーズ3: 意識の舞台・思考モジュール
### 3-1 conscious_agent.py
#### タスク
- `ConsciousAgent` クラスを実装する。
- system prompt に valence / arousal / DB1 / DB2 を埋め込む。
- `<think>...</think>` を先頭で必ず使うように促すプロンプトを実装する。
- 長期記憶との整合性評価を含める指示を入れる。
- 推論モデルの生成 API をラップする。

#### 状況
- 実装済み。

## フェーズ4: 統合チャットループ
### 4-1 main.py
#### タスク
- 入力受付、状態保持、推論、保存の一連フローを実装する。
- surprisal 計算 → emotion 更新 → dual retrieval → conscious generation → episodic 保存の順に接続する。
- 推論モデルを 4-bit 量子化でロードする分岐を入れる。
- LoRA アダプタが存在する場合は `PeftModel.from_pretrained` でアタッチする。
- モデルなしでもテスト可能な抽象層を挟む。

#### 状況
- 実装済み。

### 4-2 ランタイム整理
#### タスク
- CLI / runtime / domain logic の責務を分割する。
- 重い依存を直接呼ばずに差し替え可能にする。
- 設定値を集約する。

#### 状況
- 実装済み。

### 4-3 設定駆動化
#### タスク
- `settings.py` で runtime / model / memory / emotion / sleep / logging / paths を定義する。
- `settings.toml` から各種設定を読み込む。
- 相対パスを settings ファイル位置基準で解決する。
- CLI を `--settings` 前提で動かす。

#### 状況
- 実装済み。

## フェーズ5: 睡眠・QLoRA 学習
### 5-1 sleep_consolidation.py
#### タスク
- `SleepCycleManager` を実装する。
- 高情動エピソードを `Arousal > 0.7` または `|Valence| > 0.6` で抽出する。
- Dream Generation 用の LLM 呼び出しを実装する。
- `dream_dataset.jsonl` を生成する。
- JSONL スキーマ `{input, thought, output}` を固定する。

#### 状況
- 実装済み。

### 5-2a QLoRA 準備
#### タスク
- QLoRA 用の学習設定を `settings.toml` に追加する。
- 学習データとアダプタ保存先を settings で管理する。
- 4-bit ロード時の `nf4` / `bfloat16` 前提を設定に持たせる。
- 学習に使うベースモデル名を settings に統合する。

#### 完了条件
- settings から学習関連設定を読み出せる。
- 学習用パスを settings 基準で解決できる。

### 5-2b データ整形
#### タスク
- 学習データを `ユーザー / <think> / 応答` 形式へ変換する。
- `dream_dataset.jsonl` の各行を学習フォーマットへ結合する。
- 空データや欠損値を安全に扱う。

#### 完了条件
- 生成した JSONL から学習プロンプト文字列を復元できる。
- 1件のダミーデータでも崩れない。

### 5-2c 学習ループ
#### タスク
- 4-bit `nf4` / `bfloat16` ロードを実装する。
- `peft.prepare_model_for_kbit_training` を適用する。
- `trl.SFTTrainer` を組み込む。
- 指定ハイパーパラメータを反映する。
- 学習後に `./kagya_subjective_adapter` へ保存する。

#### 完了条件
- 小規模ダミーデータで学習が最後まで回る。
- アダプタ保存先が生成される。
- 学習完了後にアダプタの存在確認ができる。

## フェーズ6: 仕上げ
### タスク
- README を実装内容に合わせて更新する。
- 仕様と実装の差分を洗い出す。
- エラーハンドリングを整理する。
- ログ出力を最低限追加する。
- 最終的に `just check-all` を通す。
- settings ファイル込みの起動テストを追加する。

### 完了条件
- `just lint` が通る。
- `just typecheck` が通る。
- `just test` が通る。
- `just check-all` が通る。
- settings ベースの CLI テストが通る。

## 推奨実装順
1. `emotion_engine.py`
2. `surprisal_calculator.py`
3. `dual_memory_system.py`
4. `conscious_agent.py`
5. `main.py`
6. `settings.py` / `cli.py` の設定統合
7. `sleep_consolidation.py`
8. テスト拡充と全体検証

## リスクと対策
- モデルが重い: ダミーモデルとモックで先に検証する。
- ChromaDB が不安定: 永続化先を分けてテストする。
- QLoRA が環境依存: 小規模データと低ステップで検証する。
- プロンプトが肥大化する: 文字列生成を関数分割する。
- 設定項目が増える: `settings.toml` と `settings.py` を同期して管理する。
