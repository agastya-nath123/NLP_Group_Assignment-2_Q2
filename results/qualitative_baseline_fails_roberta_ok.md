# BiGRU+attention fails, RoBERTa succeeds (test set, seed 42)

### Example 1  (id `en_0544486`)

- **True rating:** 5★
- **BiGRU+attention:** 1★ (p = 0.32)
- **RoBERTa:** 5★ (p = 0.94)
- **Phenomena (auto-tagged):** negation, sarcasm cue, polarity mismatch
- **Starting points for your hypothesis (auto-generated):**
  - *negation*: Negation reverses nearby sentiment words; the BiGRU has to carry 'not' forward to the word it modifies, while RoBERTa's self-attention lets every token see the negator directly.
  - *sarcasm cue*: Possible sarcasm / irony (heuristic flag - confirm by reading): positive surface words with a negative meaning.
  - *polarity mismatch*: The words' surface polarity disagrees with the rating (e.g. only positive words but 1-2 stars), so lexical cues mislead.

> NO PUFFING UP OR LOOSE SNAGS OF FABRIC AFTER WASH!!! I bought two sets. In White. One came with a couple light brown stains. Maybe dirt. Who knows. The lining plastic package was torn. Washed them immediately after arrival with TIDE & dirt removed! So I didn’t have to return it! The fabric does NOT puff up. Looks brand new even after wash! VERY SOFT! Wrinkle free as noted!!! Very white after washing with plain blue Tide detergent. Will buy again!!! AND AGAIN!! THANK YOU!!!

### Example 2  (id `en_0600154`)

- **True rating:** 5★
- **BiGRU+attention:** 1★ (p = 0.49)
- **RoBERTa:** 5★ (p = 0.89)
- **Phenomena (auto-tagged):** contrast, sarcasm cue, polarity mismatch
- **Starting points for your hypothesis (auto-generated):**
  - *contrast*: The review pivots ('but', 'however', ...); the star rating usually follows one side of the pivot, so the model must weigh the clauses rather than count sentiment words.
  - *sarcasm cue*: Possible sarcasm / irony (heuristic flag - confirm by reading): positive surface words with a negative meaning.
  - *polarity mismatch*: The words' surface polarity disagrees with the rating (e.g. only positive words but 1-2 stars), so lexical cues mislead.

> I don’t have cravings after taking this at night! I’m horrible with eating late at night, I do it almost every night after dinner and I’m sure it’s part of the reason I’m overweight. But when I take this, the cravings are gone!!

### Example 3  (id `en_0327365`)

- **True rating:** 5★
- **BiGRU+attention:** 1★ (p = 0.35)
- **RoBERTa:** 5★ (p = 0.36)
- **Phenomena (auto-tagged):** mixed sentiment
- **Starting points for your hypothesis (auto-generated):**
  - *mixed sentiment*: Positive and negative words co-occur; middle ratings (2-4 stars) depend on their balance.

> Grimm. I loved this show and was so disappointed when it was cancelled after six seasons.

### Example 4  (id `en_0993971`)

- **True rating:** 5★
- **BiGRU+attention:** 2★ (p = 0.32)
- **RoBERTa:** 5★ (p = 0.59)
- **Phenomena (auto-tagged):** negation, sarcasm cue
- **Starting points for your hypothesis (auto-generated):**
  - *negation*: Negation reverses nearby sentiment words; the BiGRU has to carry 'not' forward to the word it modifies, while RoBERTa's self-attention lets every token see the negator directly.
  - *sarcasm cue*: Possible sarcasm / irony (heuristic flag - confirm by reading): positive surface words with a negative meaning.

> A Very subtle and male-type Fragrance !! Would Not Use any other Cologne

### Example 5  (id `en_0544268`)

- **True rating:** 1★
- **BiGRU+attention:** 3★ (p = 0.42)
- **RoBERTa:** 1★ (p = 0.65)
- **Phenomena (auto-tagged):** polarity mismatch
- **Starting points for your hypothesis (auto-generated):**
  - *polarity mismatch*: The words' surface polarity disagrees with the rating (e.g. only positive words but 1-2 stars), so lexical cues mislead.

> My other feeders work great. Poorly made. Doest attract any birds. My other feeders work great.

### Example 6  (id `en_0494424`)

- **True rating:** 4★
- **BiGRU+attention:** 3★ (p = 0.47)
- **RoBERTa:** 4★ (p = 0.86)
- **Phenomena (auto-tagged):** negation, contrast, long (>150 words)
- **Starting points for your hypothesis (auto-generated):**
  - *negation*: Negation reverses nearby sentiment words; the BiGRU has to carry 'not' forward to the word it modifies, while RoBERTa's self-attention lets every token see the negator directly.
  - *contrast*: The review pivots ('but', 'however', ...); the star rating usually follows one side of the pivot, so the model must weigh the clauses rather than count sentiment words.
  - *long (>150 words)*: Long review: the baseline truncates at 200 tokens and squeezes everything into one attention-pooled vector; RoBERTa sees up to 256 sub-words with full self-attention.

> A great product but sloppy bushings. This thing really works to lessen the impacts. Definitely makes a hard tail bike ride much nicer. It is well made, except that there is a little more play in the bushings that I think is needed. As a result, the seat can actually rock side to side a little. This does not bother me at all when riding, and may actually be a benefit, but I worry that it will get worse as it wears. Regardless, what this seat post does is transform your hard tail bike into something you’ll will want to ride more, and it makes the bike more capable. I currently have it on a fat bike. I have another good full suspension trail bike, but I doubt that I’ll buy another FS bike, ‘cause you can get a hard tail and still have comfort if equipped with this post.

### Example 7  (id `en_0079787`)

- **True rating:** 5★
- **BiGRU+attention:** 3★ (p = 0.30)
- **RoBERTa:** 5★ (p = 0.62)
- **Phenomena (auto-tagged):** OOV-heavy
- **Starting points for your hypothesis (auto-generated):**
  - *OOV-heavy*: Several words are <unk> for the baseline (misspellings, rare words); RoBERTa's byte-level BPE never produces unknown tokens, it splits them into sub-words.

> Opinion on fangirl. This novel took my heart. If you’ve read Eleanor and Park you know your in for a whirlwind. It was everything I expected.

### Example 8  (id `en_0640800`)

- **True rating:** 4★
- **BiGRU+attention:** 3★ (p = 0.45)
- **RoBERTa:** 4★ (p = 0.59)
- **Phenomena (auto-tagged):** negation, very short (<=8 words), polarity mismatch
- **Starting points for your hypothesis (auto-generated):**
  - *negation*: Negation reverses nearby sentiment words; the BiGRU has to carry 'not' forward to the word it modifies, while RoBERTa's self-attention lets every token see the negator directly.
  - *very short (<=8 words)*: Very little evidence; a single word decides the prediction.
  - *polarity mismatch*: The words' surface polarity disagrees with the rating (e.g. only positive words but 1-2 stars), so lexical cues mislead.

> Not bad for the price
