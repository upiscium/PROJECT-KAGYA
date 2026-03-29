# PROJECT-KAGYA 実装チェックリスト

## 1. 土台
- [x] `src/project_kagya/` を作成する
- [x] `src/project_kagya/__init__.py` を追加する
- [x] `tests/` を作成する
- [x] 共通設定・型定義の配置方針を決める
- [x] `pyproject.toml` に必要依存があるか確認する
- [x] 既存 CLI エントリポイントを統合前提で保持する
- [x] `settings.toml` をプロジェクトの単一設定ソースとして追加する

## 2. 数式系コア
- [x] `emotion_engine.py` に `EmotionEngineAllostasis` を実装する
- [x] `optimal_loss` と `adaptation_rate` を保持する
- [x] Arousal 更新式を実装する
- [x] Wundt 曲線の valence 変換を実装する
- [x] Valence 更新式を実装する
- [x] `optimal_loss` の移動平均更新を実装する
- [x] 値域クランプを実装する
- [x] 数式検証の単体テストを書く
- [x] `surprisal_calculator.py` を作成する
- [x] tokenizer / model を受け取る loss 計算関数を定義する
- [x] `context_text` 部分を `-100` でマスクする
- [x] `torch.no_grad()` で実行する
- [x] context と新規入力を結合するヘルパーを作る
- [x] ダミーモデルで loss 計算のテストを書く

## 3. 記憶層
- [x] `dual_memory_system.py` を作成する
- [x] ChromaDB クライアント初期化を実装する
- [x] `hippocampus` コレクションを管理する
- [x] `cortex` コレクションを管理する
- [x] episodic 保存スキーマを定義する
- [x] `save_episodic()` を実装する
- [x] 検索結果整形ヘルパーを実装する
- [x] `retrieve_context()` を実装する
- [x] DB1 / DB2 の検索結果を統合フォーマットで返す
- [x] 空検索時の安全な挙動を確認する
- [x] `consolidate_to_semantic()` を実装する
- [x] DB1 の生ログを要約・事実抽出して DB2 に保存する
- [x] 移行済みレコードを DB1 から削除する
- [x] 記憶保存・検索・統合のテストを書く

## 4. 思考エージェント
- [x] `conscious_agent.py` を作成する
- [x] `ConsciousAgent` クラスを実装する
- [x] system prompt に valence を含める
- [x] system prompt に arousal を含める
- [x] system prompt に DB1 の検索結果を含める
- [x] system prompt に DB2 の検索結果を含める
- [x] `<think>...</think>` を先頭に出す指示を入れる
- [x] 長期記憶との整合性評価を含める
- [x] 生成 API のラッパーを実装する
- [x] プロンプト構築のテストを書く

## 5. 統合チャットループ
- [x] `main.py` を作成する
- [x] モック差し替え可能な設計にする
- [x] 入力受付を実装する
- [x] surprisal 計算を接続する
- [x] emotion 更新を接続する
- [x] dual retrieval を接続する
- [x] conscious generation を接続する
- [x] episodic 保存を接続する
- [x] 処理順序を固定する
- [x] 推論モデルの 4-bit ロード分岐を実装する
- [x] LoRA アダプタ有無の分岐を実装する
- [x] `PeftModel.from_pretrained` でアタッチする経路を実装する
- [x] 統合フローのテストを書く
- [x] CLI を `settings.toml` 前提に寄せる
- [x] ログと睡眠出力の設定反映を実装する

## 6. 睡眠・QLoRA 学習
- [x] `sleep_consolidation.py` を作成する
- [x] `SleepCycleManager` を実装する
- [x] 高情動エピソード抽出条件を実装する
- [x] Dream Generation 用の LLM 呼び出しを実装する
- [x] `dream_dataset.jsonl` を生成する
- [x] JSONL スキーマを固定する
- [x] 生成データのテストを書く
- [ ] QLoRA 用の学習設定を `settings.toml` に追加する
- [ ] `sleep` 配下で学習データの出力先を設定化する
- [ ] 4-bit `nf4` / `bfloat16` ロードを実装する
- [ ] `prepare_model_for_kbit_training` を適用する
- [ ] `trl.SFTTrainer` を組み込む
- [ ] 学習データを `ユーザー / <think> / 応答` 形式に整形する
- [ ] 指定ハイパーパラメータを反映する
- [ ] 学習結果の保存先を `settings.toml` で管理する
- [ ] `./kagya_subjective_adapter` へ保存する
- [ ] 小規模ダミーデータで学習完了テストを書く
- [ ] 学習完了後にアダプタ存在確認を行う

## 7. 仕上げと検証
- [x] README を実装内容に合わせて更新する
- [x] 例外処理を見直す
- [x] ログ出力を最低限追加する
- [x] 仕様との差分を確認する
- [x] `just lint` を通す
- [x] `just typecheck` を通す
- [x] `just test` を通す
- [x] `just check-all` を通す
- [x] settings ファイル込みの起動テストを追加する
