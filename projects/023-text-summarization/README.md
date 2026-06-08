# News Summarization with T5, PEGASUS and BART

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Abstractive%20Summarization-green)

## Overview
Abstractive news summarization that generates concise summaries in new words rather than extracting source sentences. The project fine-tunes and compares three transformer encoder-decoder models — T5, PEGASUS and BART — on a news article / summary dataset.

## Dataset
- [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) (`sunnysai12345/news-summary`, ~51 MB)
- Columns used: `headlines`, `text` (short summary), `ctext` (full article). The notebook builds an input field `news = headlines + ". " + ctext` and targets `summary = text`.
- Rows with `ctext_length < text_length` are dropped; an 80/20 train/val split is taken after holding out 1% as a test set.
- Not committed here. Download with:
  ```bash
  kaggle datasets download sunnysai12345/news-summary
  ```

## Approach
1. **Preprocessing**: lowercase all text, drop NaNs, build the combined `news` input field, prepend the `"summarize: "` task prefix, convert to a Hugging Face `DatasetDict`.
2. **Tokenization**: source truncated to `max_length=512`, target labels to `max_length=128`; dynamic padding via `DataCollatorForSeq2Seq`.
3. **Models fine-tuned** (all `AutoModelForSeq2SeqLM` + `Seq2SeqTrainer`):
   - **T5** (`t5-base`): lr 2e-5, batch 16, 4 epochs, fp16.
   - **PEGASUS** (`google/pegasus-large`): lr 1e-6, batch 8, 3 epochs, fp32 (fp16 unsupported).
   - **BART** (`facebook/bart-large-cnn`): lr 1e-6, batch 8, 3 epochs, fp16.
4. **Metric**: ROUGE (via `rouge.Rouge`) computed in `compute_metrics` with `predict_with_generate=True`.
5. **Inference**: each fine-tuned model is reloaded and run through a `transformers` summarization `pipeline` on held-out test articles.

## Results
The notebook computes **ROUGE** scores (ROUGE-1/2/L precision, recall, F1) during evaluation, but this folder has no executed run log and the notebook outputs are not committed. The author's qualitative conclusion was a relative compute-cost ranking: `bart-large-cnn ≈ 1.5 × pegasus-large ≈ 2 × t5-base`.

Run the notebook to reproduce the metrics (ROUGE-1/2/L).

| Model | Metric reported |
|-------|-----------------|
| T5 (t5-base) | ROUGE-1 / ROUGE-2 / ROUGE-L |
| PEGASUS (pegasus-large) | ROUGE-1 / ROUGE-2 / ROUGE-L |
| BART (bart-large-cnn) | ROUGE-1 / ROUGE-2 / ROUGE-L |

## Key Takeaways
- A single text-to-text framing (with a task prefix) lets all three architectures share the same data pipeline.
- BART and PEGASUS are pre-trained with summarization-aligned objectives, so they are expected to need fewer epochs / lower learning rates than generic T5.
- GPU memory drives the design: fp16 used where supported, smaller batch sizes for the larger models, and `torch.cuda.empty_cache()` between models.
- Possible improvements: report ROUGE side-by-side, add beam-search decoding parameters, and evaluate on the full test set rather than 5 samples.

## How to Run
```bash
pip install torch transformers datasets lightning rouge pandas numpy seaborn matplotlib
# place news_summary.csv in the working directory (see Dataset)
jupyter notebook notebook.ipynb
```
Training the large models requires a GPU (the author used an NVIDIA L4 24 GB).

## Credit
> Based on ["News Summarization - T5, PEGASUS and BART"](https://www.kaggle.com/code/zifengjiang/news-summarization-t5-pegasus-and-bart) by Zifeng Jiang (Andy) on Kaggle (46 votes).
> Dataset: [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary). Adapted and documented for this portfolio.
