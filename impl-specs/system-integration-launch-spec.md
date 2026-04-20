# System Integration Launch Spec

## 目的
`PROJECT-KAGYA` の各モジュールをひとつの起動点から連携させ、入力受理、推論、記憶、品質検査、睡眠統合、学習までを順番に実行できるようにする。

## 現状
- `project-kagya` の現在の CLI は `project_kagya.cli:main` で、実体は `embodied_emotion` のデモ起動に固定されている。
- `multimodal_fastapi_interface.py` は API 入口を提供するが、内部の推論・記憶・学習にはまだ接続していない。
- `sleep_consolidation.py` と `sleep_consolidation_training.py` は JSONL ベースの後段パイプラインとして独立している。

## 位置づけ
- 受信層: `multimodal_fastapi_interface.py`
- 覚醒時推論: `emotion_engine.py`, `dual_memory_system.py`, `conscious_agent.py`
- 品質検査: `thought_quality_assurance.py`, `data_quality_evaluation.py`, `drift_control.py`
- 睡眠統合: `sleep_consolidation.py`
- 学習: `sleep_consolidation_training.py`, `qlora_training.py`

## 公開インターフェース

### `project_kagya.cli:main`
- 単一の起動点とする。
- モード引数を受け、各フェーズを明示的に切り替える。

### 推奨モード
- `--serve`: FastAPI を起動する。
- `--demo`: 既存の埋め込み感情デモを実行する。
- `--consolidate`: 睡眠統合を実行する。
- `--train`: JSONL から学習を実行する。
- `--pipeline full`: 受信から学習までを順に実行する。

## 起動フロー
1. `settings.toml` を読む。
2. 受信層またはデモを起動する。
3. 入力を正規化する。
4. `EmotionState` を更新する。
5. `DualMemorySystem` から文脈を引く。
6. `ConsciousAgent` で応答を生成する。
7. `ThoughtQualityAssurer` と `DataQualityEvaluator` で例を検査する。
8. `DriftController` で更新採否を判断する。
9. `SleepCycleManager` で JSONL を生成する。
10. `SleepConsolidationTrainingPipeline` で学習する。
11. `QLoRATrainer` でアダプタを保存・再読込する。

## データ契約
- 学習用 JSONL は 1 行 1 例とする。
- 各行は `input`, `thought`, `output`, `source_ids`, `confidence`, `status` を持つ。
- `status == "confirmed"` かつ `thought` が空でない例のみ学習対象とする。
- `confidence` は閾値以上のものだけ学習に回す。

## モード別責務

### `--serve`
- FastAPI アプリを返す。
- `/ingest`, `/chat`, `/stream` を公開する。

### `--pipeline full`
- 受信、推論、品質検査、睡眠統合、学習を順番に実行する。
- 途中失敗時はどのフェーズで落ちたかを明示する。

### `--consolidate`
- 睡眠統合だけを実行する。
- 出力先は `settings.toml` の `sleep.dream_dataset_path` を使う。

### `--train`
- JSONL を入力として学習だけを実行する。
- 学習実体は注入された backend / trainer に委譲する。

## 実装上の制約
- FastAPI と学習処理は分離する。
- `settings.toml` を単一の設定元にする。
- 既存のモジュールを大きく書き換えず、薄いオーケストレーション層を追加する。
- `project-kagya` の既存デモ動作は `--demo` に残す。

## エラー時の挙動
- 設定が読めない場合は即失敗する。
- 空の学習データセットは正常終了とする。
- 無効な行はスキップし、全体処理はできるだけ継続する。
- 学習失敗時は中間生成物を破棄しない。

## 仕様テスト観点
- CLI がモードを分岐できること。
- FastAPI 入口が生成できること。
- `ConsciousAgent` が記憶文脈を含む prompt を作ること。
- 睡眠統合が JSONL を生成すること。
- 学習パイプラインが空データセットで安全に終了すること。
- 学習済みアダプタを再読込できること。

## 実行例
```bash
uv run project-kagya
uv run project-kagya --serve
uv run project-kagya --consolidate
uv run project-kagya --train
uv run project-kagya --pipeline full
```

## 運用メモ
- `settings.toml` の `sleep.dream_dataset_path` が睡眠統合と学習の接続点になる。
- `--pipeline full` は小規模な統合検証向けで、本番では必要なモードだけを個別に使う。
- FastAPI 起動と学習実行は同時起動しない。

## 非目標
- モデル品質の最適化
- 分散学習
- 永続バックエンドの導入
- 新しい記憶理論の追加
