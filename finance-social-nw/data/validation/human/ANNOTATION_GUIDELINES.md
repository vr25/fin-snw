# Annotation Guidelines: Relations Between Companies in AI News

Thank you for helping. You will read short excerpts from news articles. Each excerpt mentions two
companies, and you will record **what relationship the article asserts between those two companies**.

- **File to fill in:** `annotation_sheet.csv` (159 rows). Open it in Excel, Google Sheets or Numbers.
- **Time needed:** about 1–1.5 minutes per row, 3–4 hours in total. You can split this across sessions.
- **Please work independently.** Do not discuss items with other annotators, and **do not use ChatGPT
  or any other AI tool**. The purpose is to get *human* judgements to compare against AI labels.

---

## 1. What each row contains

| Column | Meaning |
|---|---|
| `row` | Row number (the order is random) |
| `item_id` | Internal ID. Do not edit |
| `ticker_a`, `company_a`, `ticker_b`, `company_b` | The two companies to judge |
| `date` | Publication date |
| `url` | Link to the original article (use it if the excerpt is not enough) |
| `full_text_file` | Local copy of the article text, if you have the project folder |
| `excerpt` | The sentences that mention either company. Sentences starting with `>>` mention **both** companies. `[...]` marks skipped text |

**Fill in these five columns:**

| Column | What to enter |
|---|---|
| `relation` | Exactly one of: `causal-driver`, `substitute-comparison`, `co-hype`, `unrelated-mention` |
| `sentiment_alignment` | A number from `-1` to `1` (see Section 4) |
| `confidence` | `1` = guess, `2` = fairly sure, `3` = certain |
| `needed_full_text` | `yes` if you had to open the URL or full text, otherwise leave blank |
| `notes` | Optional. Anything unusual (for example "garbled text" or "wrong company") |

---

## 2. The four relation labels

Judge **only what the article says about THIS pair** (company_a and company_b). Ignore other companies
in the article, and ignore what you personally know about the companies.

### `causal-driver`: one company's business directly affects the other's
The article states or clearly implies a direct business link. For example:
- supplier and customer ("Microsoft is buying tens of thousands of Nvidia GPUs")
- one company uses the other's products or services
- a partnership, joint venture or investment between them
- one company's demand, results or decisions driving the other's ("Nvidia's revenue surged on
  orders from cloud providers such as Google")

### `substitute-comparison`: the two are rivals, or are explicitly compared
- The article presents them as competitors ("Google's Gemini is its answer to Microsoft-backed OpenAI").
- The article explicitly compares them on products, market share, valuation or stock performance
  ("AMD's MI300 is positioned as an alternative to Nvidia's H100"; "Meta outperformed Alphabet this year").

### `co-hype`: mentioned together as part of a shared trend, with no direct link
- Both are cast as participants in, or beneficiaries of, a common theme such as the AI boom, the
  "Magnificent Seven", big tech earnings season or a regulatory crackdown.
- Example: "AI stocks such as Nvidia and Microsoft rallied on Tuesday."
- There is no direct business link and no explicit comparison between the two.

### `unrelated-mention`: both appear, but no relationship is asserted
- incidental mentions, lists, boilerplate, "related articles" links, stock tickers in a sidebar
- the article is off-topic (for example about a restaurant) and the firms appear only in passing
- the text is garbled, unreadable, or not really an article

---

## 3. Decision rules

**If more than one label applies, choose the first one that fits, in this order:**

> 1. `causal-driver` → 2. `substitute-comparison` → 3. `co-hype` → 4. `unrelated-mention`

So an article saying "Microsoft buys Nvidia chips, and both stocks rose in the AI rally" is
`causal-driver`, because the direct link comes first.

**Quick decision path:**
1. Is there a direct business link between the two companies (buyer/supplier, partner, investor, one
   driving the other)? → `causal-driver`
2. Otherwise, are they presented as competitors or explicitly compared? → `substitute-comparison`
3. Otherwise, are they grouped together as part of a common trend or theme? → `co-hype`
4. Otherwise → `unrelated-mention`

**Tricky cases:**

| Situation | Label |
|---|---|
| Both named in a list of "AI winners" or "Big Tech stocks" | `co-hype` |
| Both named in a list with no theme (for example "companies mentioned in this article: ...") | `unrelated-mention` |
| Both face the same regulation or lawsuit, and the article treats them as a group | `co-hype` |
| Both face the same regulation, and the article contrasts how they responded | `substitute-comparison` |
| "X is racing Y to build AI models" | `substitute-comparison` |
| "Y invested in OpenAI, which competes with X" (a link only through a third party) | `substitute-comparison` if they are framed as rivals, otherwise `co-hype` |
| One firm is a customer of the other, mentioned only in passing | still `causal-driver` |
| "Meta" used in a non-company sense, or a different firm with a similar name | `unrelated-mention` (add a note) |
| The excerpt is unclear, but the full article is clear | Use the full article and write `yes` in `needed_full_text` |

The excerpt is usually enough. If no sentence starts with `>>` (the two companies are never named in
the same sentence), check the full article before choosing `unrelated-mention`.

---

## 4. Sentiment alignment (−1 to 1)

This asks whether the article's tone about the two companies points **in the same direction**.

| Value | Meaning |
|---|---|
| `1` | Clearly the same direction (both positive, or both negative) |
| `0.5` | Mostly the same direction |
| `0` | Neutral, mixed, or no sentiment expressed |
| `-0.5` | Mostly opposite directions |
| `-1` | Clearly opposite (one wins while the other loses) |

Examples: "Nvidia and Microsoft both hit record highs" → `1`. "AMD gained share at Nvidia's expense" → `-1`.
A factual list with no tone → `0`.

---

## 5. Procedure

1. Read Sections 2–4 once before starting. Keep this page open while you work.
2. Work through the rows in order. For each row, read the excerpt, then fill in `relation`,
   `sentiment_alignment` and `confidence`.
3. Type labels **exactly** as spelled above (lower case, with hyphens). A dropdown or copy-paste helps.
4. Do not change any other column, and do not reorder or delete rows.
5. Save as CSV (UTF-8) named `annotation_sheet_<your initials>.csv` and send it back.

If two people annotate, each works alone on a separate copy. Disagreements are resolved afterwards.

---

*For the researcher (not needed by annotators):* `key.json` holds the model labels and sampling
strata. Do not share it with annotators. After collecting sheets, run
`python scripts/score_human_labels.py data/validation/human/annotation_sheet_*.csv`.
