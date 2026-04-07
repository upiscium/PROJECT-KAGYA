# DualMemorySystem 仕様メモ

## 目的
短期エピソード記憶 (`DB1`) と長期意味記憶 (`DB2`) を分離し、会話の直近文脈と確定済み事実を安全に扱う。

## 公開インターフェース

### `save_episodic(user_input, response, valence, arousal)`
- `DB1` に1件のエピソードを保存する。
- 保存内容は少なくとも `user_input`, `response`, `valence`, `arousal`, `timestamp`, `source` を含む。
- 引数は次の意味を持つ。
  - `user_input`: ユーザー発話の原文
  - `response`: 直前の応答原文
  - `valence`: `-1.0` から `1.0` の範囲の快不快値
  - `arousal`: `0.0` から `1.0` の範囲の覚醒度
- 返り値は保存結果を表す識別子または保存メタ情報とする。
- 範囲外の `valence` / `arousal` は保存前にクランプする。

### `retrieve_context(query)`
- `DB1` と `DB2` の両方から検索する。
- 返却値はプロンプト挿入用の整形済み文字列とする。
- `DB1` の結果を優先し、`DB2` は補助情報として扱う。
- 返却文字列には少なくとも以下の区画を含める。
  - `Recent Memory` / `DB1`
  - `Semantic Memory` / `DB2`
  - `Priority Notes`
- 返却文字列はそのままシステムプロンプトに挿入できる形式とする。
- 空の検索結果では、区画は残しつつ「なし」と明示する。

#### 返却フォーマット例
```text
Recent Memory (DB1)
- [high] 2026-04-07 user prefers concise answers
- [medium] 2026-04-06 user asked about memory safety

Semantic Memory (DB2)
- [confirmed] user prefers Japanese replies
- [tentative] user is exploring memory-centric agents

Priority Notes
- DB1 first for recency
- DB2 only for confirmed facts
```

### `consolidate_to_semantic(llm_pipeline)`
- `DB1` のエピソードから、`DB2` に昇格させる候補を抽出する。
- 抽出結果は構造化された事実候補として保存する。
- 保存成功前に `DB1` の元データを削除してはならない。
- 返り値は、移行件数・保留件数・失敗件数を含む集計情報とする。
- `llm_pipeline` は抽出専用の依存であり、記憶操作本体は外部から差し替え可能にする。

#### 状態遷移
1. `DB1` から候補エピソードを抽出する。
2. `llm_pipeline` で事実候補を生成する。
3. 候補ごとに `confirmed`, `tentative`, `conflicted` のいずれかに分類する。
4. `confirmed` のみ `DB2` に保存する。
5. `tentative` は保留として残す。
6. `conflicted` は保存せず、既存の関連記憶に矛盾フラグを付ける。
7. `DB2` 保存に成功したものだけ、対応する `DB1` の元データをアーカイブ後に削除する。

#### 判定基準
- `confirmed`: 明示的・反復的・一貫的で、永続化に値する。
- `tentative`: 有望だが、確認が足りない。
- `conflicted`: 既存の確定済み記憶と衝突する。

## データ形式

### Episodic record
```json
{
  "id": "str",
  "user_input": "str",
  "response": "str",
  "valence": 0.0,
  "arousal": 0.0,
  "timestamp": "iso-8601",
  "source": "chat",
  "confidence": 1.0
}
```

### Semantic record
```json
{
  "id": "str",
  "fact": "str",
  "source_ids": ["str"],
  "confidence": 0.0,
  "status": "confirmed|tentative|conflicted",
  "timestamp": "iso-8601"
}
```

## 保存ルール
- `DB2` に保存できるのは、次のいずれかに限定する。
  - ユーザーが明示した永続的嗜好
  - 繰り返し確認された事実
  - 明示的なプロフィール情報
- 単発の雑談、曖昧な推測、感情的な発話は `DB2` に昇格しない。
- 各記憶には `confidence` と `source` を付与する。

## 移行ルール
- `DB1 -> DB2` は二段階で行う。
  1. 仮保存
  2. 次回以降の整合確認後に正式保存
- 失敗時に復旧できるよう、元データはアーカイブ期間を持つ。
- `DB1` の削除は、`DB2` への正式保存完了後にのみ行う。

## 検索ルール
- `DB1` は最近の会話と情動メタデータを優先する。
- `DB2` は確定済み事実のみを返す。
- 両者の結果を結合する際は、優先度を明示する。
- 類似度だけでなく、時系列と信頼度でも順位付けする。

## 矛盾処理
- 同じ属性に異なる値が複数存在する場合は矛盾として扱う。
- 矛盾した記憶は自動降格または保留状態にする。
- 矛盾解消前の `DB2` 反映は行わない。

## エラー時の挙動
- `llm_pipeline` が失敗した場合、移行は中断し `DB1` を保持する。
- `DB2` 保存に失敗した場合、`DB1` 側の正式削除は行わない。
- 検索対象が空の場合、空文字ではなく「記憶なし」を示す明示的な結果を返す。

## 仕様テスト観点
- `save_episodic` が必要メタデータ付きで `DB1` に保存すること
- `retrieve_context` が `DB1` 優先で結合すること
- `consolidate_to_semantic` が成功前削除をしないこと
- 単発の雑談が `DB2` に昇格しないこと
- 矛盾する記憶が降格扱いになること
- `retrieve_context` が空時でもフォーマットを崩さないこと
- `save_episodic` が範囲外の値をクランプすること
- `consolidate_to_semantic` の集計が移行・保留・失敗を分けること

## 非目標
- 完全な自律学習
- 事実抽出の100%正確性
- すべての会話を長期保存すること
