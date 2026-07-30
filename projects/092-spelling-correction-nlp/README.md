# Spelling Correction System

![Domain](https://img.shields.io/badge/domain-NLP-blue)
![Task](https://img.shields.io/badge/task-Auto--correction-green)

## Overview
An unsupervised auto-correct system that detects misspelled words and substitutes the
most probable correction. It builds a word-frequency probability model from a text corpus
and generates candidate corrections via single- and double-edit operations
(insert / delete / switch / replace), then ranks candidates by corpus probability.

## Dataset
- A plain-text spelling corpus (`Spellings_data.txt`, read with ISO-8859-1 encoding) used
  to build the vocabulary and word-probability distribution.
- The dataset reference in `source.json` is unresolved (no concrete Kaggle ref/url), and no
  `data/` folder is committed. Supply any reasonably large English text corpus and point
  the loader at it (the notebook expects `../input/spellings-data/Spellings_data.txt`).
- From the captured run: the corpus yields **58,110 unique words** in the vocabulary.

## Approach
1. **Process corpus** — `process_data()` tokenizes with `re.findall(r'\w+', ...)` and
   lowercases; build `vocab = set(words)` and a `Counter` frequency dictionary.
2. **Filter misspells** — `find_wrong_word()` flags any token not present in `vocab`.
3. **Candidate generation** — `delete_letter`, `switch_letter`, `replace_letter`,
   `insert_letter`; combined into `edit_one_letter` and `edit_two_letters`
   (for `'at'`: 129 one-edit candidates, 7154 two-edit candidates in the run).
4. **Probability model** — `get_probs()` turns word counts into `P(word) = count/total`
   over all 58,110 entries.
5. **Correction** — for each wrong word, choose the highest-probability candidate among the
   valid edits; falls back to two-edit candidates when needed.
6. **Comparison libraries** — `textblob`, `spellchecker`, and `pyspellchecker` are also
   demonstrated on a sample paragraph for qualitative comparison.

## Results
Real values extracted from the captured run log:

| Unit test | Accuracy |
|---|---|
| Basic auto-correct system — test 1 | **65.19%** |
| Basic auto-correct system — test 2 | **66.25%** |

Qualitative examples from the run:
- `"honsty is the best pooliccy"` → `"Honesty is the best policy"`
- `"ffrront"` → `"Front"`

## Key Takeaways
- A pure edit-distance + corpus-frequency model already corrects ~65% of test cases.
- Errors come from candidates that are valid words but wrong in context (no context model),
  e.g. `goiing` → `goring` instead of `going`.
- Adding an n-gram language model or contextual scoring would improve over single-word
  probability ranking.

## How to Run
```bash
pip install nltk numpy pandas textblob spellchecker pyspellchecker
python -c "import nltk; nltk.download('punkt')"
jupyter notebook notebook.ipynb
```
Provide a text corpus at `../input/spellings-data/Spellings_data.txt` (or edit the path).

## Credit
> Based on ["Spelling AutoCorrection system"](https://www.kaggle.com/code/jayantyadav2003/spelling-autocorrection-system) by Jayant Yadav on Kaggle (11 votes).
> Dataset: a public English text corpus (the original kernel's dataset ref is unspecified). Adapted and documented for this portfolio.
