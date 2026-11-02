# RoBERTa fails, BiGRU+attention succeeds (test set, seed 42)

### Example 1  (id `en_0540577`)

- **True rating:** 3★
- **BiGRU+attention:** 3★ (p = 0.31)
- **RoBERTa:** 1★ (p = 0.38)
- **Phenomena (auto-tagged):** negation
- **Starting points for your hypothesis (auto-generated):**
  - *negation*: Negation reverses nearby sentiment words; the BiGRU has to carry 'not' forward to the word it modifies, while RoBERTa's self-attention lets every token see the negator directly.

> Must not have bought the right style. Doesn't fit my spout must be to modern for it.

### Example 2  (id `en_0294586`)

- **True rating:** 1★
- **BiGRU+attention:** 1★ (p = 0.35)
- **RoBERTa:** 3★ (p = 0.42)
- **Phenomena (auto-tagged):** contrast, mixed sentiment
- **Starting points for your hypothesis (auto-generated):**
  - *contrast*: The review pivots ('but', 'however', ...); the star rating usually follows one side of the pivot, so the model must weigh the clauses rather than count sentiment words.
  - *mixed sentiment*: Positive and negative words co-occur; middle ratings (2-4 stars) depend on their balance.

> Broken bottle. Love the product but one was broken in box and left a awful mess.

### Example 3  (id `en_0103134`)

- **True rating:** 1★
- **BiGRU+attention:** 1★ (p = 0.30)
- **RoBERTa:** 3★ (p = 0.29)
- **Phenomena (auto-tagged):** negation, mixed sentiment
- **Starting points for your hypothesis (auto-generated):**
  - *negation*: Negation reverses nearby sentiment words; the BiGRU has to carry 'not' forward to the word it modifies, while RoBERTa's self-attention lets every token see the negator directly.
  - *mixed sentiment*: Positive and negative words co-occur; middle ratings (2-4 stars) depend on their balance.

> Product works. This review is not for the product its for the seller. There was a mistake on there part and I didn't receive the right product . Would have been too much of a hassle for me to return. Gave the tablet form vitamins to a friend. This product does work.

### Example 4  (id `en_0124671`)

- **True rating:** 4★
- **BiGRU+attention:** 4★ (p = 0.29)
- **RoBERTa:** 5★ (p = 0.87)
- **Phenomena (auto-tagged):** sarcasm cue
- **Starting points for your hypothesis (auto-generated):**
  - *sarcasm cue*: Possible sarcasm / irony (heuristic flag - confirm by reading): positive surface words with a negative meaning.

> buy it!! needs auto cut-off for alarm, loud alarm tone
