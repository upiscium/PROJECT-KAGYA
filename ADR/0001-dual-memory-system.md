# ADR 0001: DualMemorySystem を短期/長期の二層記憶として採用する

## Status
Accepted

## Context
`impl-specs/dual-memory-system-spec.md` では、会話の直近文脈と確定済み事実を分離して扱うために、短期エピソード記憶 (`DB1`) と長期意味記憶 (`DB2`) を持つ `DualMemorySystem` が定義されている。`DB1` には会話の生ログと情動メタデータを保持し、`DB2` には睡眠時の抽出を経た恒久的な事実・関係性を保持する。

この構成は、単一の永続メモリにすべてを入れる方式よりも、検索対象と保存対象を分離できるため、文脈汚染と長期記憶の誤保存を抑えやすい。

## Decision
`DualMemorySystem` は次の方針で実装する。

- `DB1` を短期エピソード記憶、`DB2` を長期意味記憶として分離する
- `save_episodic(user_input, response, valence, arousal)` で `DB1` に保存する
- `retrieve_context(query)` で `DB1` 優先、`DB2` 補助の検索結果を返す
- `consolidate_to_semantic(llm_pipeline)` で `DB1` から事実候補を抽出し、`DB2` に昇格する
- `DB2` への保存は `confirmed` のみとし、`tentative` と `conflicted` は分離して扱う
- 保存前のクランプ、移行成功後の削除、失敗時の保持を必須とする

## Consequences
### Positive
- 直近会話と永続事実を混ぜずに扱える
- 検索結果の役割分担が明確になる
- 移行ルールを持つことで、幻覚の長期固定を抑えやすい

### Negative
- `DB2` 昇格条件の設計と運用が必要になる
- 要約や抽出の品質が低いと、誤った事実を保存しやすい
- `DB1 -> DB2` の二段階移行により、実装と検証が少し複雑になる

## Alternatives Considered
- 単一メモリに全記録を保存する
  - 実装は簡単だが、文脈汚染と検索ノイズが大きい
- `DB2` だけを持つ
  - 長期保存はできるが、直近文脈の追跡が弱い
- `DB1` だけを持つ
  - 実装は単純だが、長期的一貫性を作りにくい

## Related Tests
- `save_episodic` が範囲外の値をクランプする
- `retrieve_context` が `DB1` 優先で整形出力する
- `consolidate_to_semantic` が `confirmed` / `tentative` / `conflicted` を分ける
- `DB1` の元データを成功前に削除しない
