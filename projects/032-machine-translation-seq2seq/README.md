# Machine Translation with Seq2Seq LSTMs

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Machine%20Translation-green)

## Overview
German-to-English machine translation using an encoder-decoder (Seq2Seq) network built from LSTMs in Keras. The encoder compresses a German sentence into fixed-dimension state vectors that initialize the decoder, which emits the English translation one word at a time.

## Dataset
- [Bilingual Sentence Pairs](https://www.kaggle.com/datasets/alincijov/bilingual-sentence-pairs) (`alincijov/bilingual-sentence-pairs`, ~449 MB) — tab-separated German/English sentence pairs (`deu.txt`).
- The full corpus has 150k+ pairs; the notebook uses the **first 50,000** pairs to keep training tractable.
- Not committed here. Download with:
  ```bash
  kaggle datasets download alincijov/bilingual-sentence-pairs
  ```

## Approach
1. **Preprocessing**: read `deu.txt`, split into sentence pairs, strip punctuation, lowercase all text.
2. **Tokenization**: separate Keras `Tokenizer` for English and German; compute per-language vocabulary sizes and max sequence lengths; integer-encode and pad sequences.
3. **Model** (`define_model`, Keras `Sequential`):
   - `Embedding(in_vocab, 512, mask_zero=True)`
   - `LSTM(512)` — encoder
   - `RepeatVector(out_timesteps)`
   - `LSTM(512, return_sequences=True)` — decoder
   - `Dense(out_vocab, activation='softmax')`
4. **Training**: optimizer `RMSprop(lr=0.001)`, loss `sparse_categorical_crossentropy`, 30 epochs, batch size 512, 20% validation split, `ModelCheckpoint` to save the best model.
5. **Inference**: greedy decode via `argmax`, map indices back to words, compare actual vs. predicted in a DataFrame.

## Results
The notebook trains with a loss curve (`history.history['loss']`) and produces a table of actual vs. predicted translations, but it does not compute a numeric translation metric (e.g. BLEU), and no run log is present in this folder.

Run the notebook to reproduce the metrics. Reported signals: training/validation `sparse_categorical_crossentropy` loss curve, plus qualitative actual-vs-predicted translation samples.

## Key Takeaways
- A compact 2-LSTM encoder-decoder with `RepeatVector` bridging is enough to learn basic translations on 50k pairs.
- `sparse_categorical_crossentropy` avoids one-hot encoding the huge target vocabulary, saving memory.
- `mask_zero=True` lets the embedding ignore padding tokens.
- Limitations: no attention mechanism (fixed-length bottleneck) and greedy decoding cap quality; adding attention/beam search and reporting BLEU are natural next steps.

## How to Run
```bash
pip install tensorflow keras numpy pandas matplotlib
# place deu.txt under ../input/bilingual-sentence-pairs/ (see Dataset)
jupyter notebook notebook.ipynb
```

## Credit
> Based on ["Machine Translation | Seq2Seq | LSTMs"](https://www.kaggle.com/code/harshjain123/machine-translation-seq2seq-lstms) by Harsh Jain on Kaggle (671 votes).
> Dataset: [Bilingual Sentence Pairs](https://www.kaggle.com/datasets/alincijov/bilingual-sentence-pairs). Adapted and documented for this portfolio.
