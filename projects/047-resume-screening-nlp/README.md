# Resume Screening with NLP

![NLP](https://img.shields.io/badge/domain-NLP-blue) ![Task](https://img.shields.io/badge/task-Document%20Matching-green)

## Overview
An NLP pipeline that screens resumes against job descriptions by ranking candidates with TF-IDF + cosine similarity. Resume PDFs are parsed and cleaned, vectorized alongside job descriptions, and the most similar resumes are surfaced per job.

## Dataset
- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) (`snehaanbhawal/resume-dataset`) — `Resume.csv` plus categorized resume PDFs under `data/`.
- [Resume and Job Description](https://www.kaggle.com/datasets/pranavvenugo/resume-and-job-description) (`pranavvenugo/resume-and-job-description`, ~57 MB) — job descriptions.
- Not committed here. Download with:
  ```bash
  kaggle datasets download snehaanbhawal/resume-dataset
  kaggle datasets download pranavvenugo/resume-and-job-description
  ```

## Approach
1. **PDF parsing**: extract text from each resume PDF with `pypdf.PdfReader` (`extract_text_from_pdf`).
2. **Preprocessing** (`preprocess_text`): lowercase, remove punctuation/numbers, tokenize, drop NLTK stopwords, and stem/normalize; strip domain-noise words (e.g. `company`, `name`, `state`, `work`, `manag`, `project`).
3. **EDA**: per-category WordClouds across the resume categories.
4. **Label encoding**: `LabelEncoder` on the `Category` column; `train_test_split` for the feature column.
5. **Vectorization & matching**: `TfidfVectorizer(stop_words='english', max_features=800)` fit on resumes; job descriptions transformed into the same space.
6. **Ranking**: `sklearn.metrics.pairwise.cosine_similarity` between each job description and resumes, returning the top-`k` (k=5) matches per job with their cosine scores.

## Results
This is an unsupervised matching/ranking task, so there is no accuracy/F1 score. The run log contains the **per-job ranked matches** with cosine-similarity scores, for example:

| jobId | resumeId | cosineSimilarity | domain |
|-------|----------|------------------|--------|
| 0 | 37660306 | 0.451 | Sales Specialist |
| 4 | 12802330 | 0.349 | Web Developer |
| 5 | 25497147 | 0.415 | Frontend Web Developer |
| 9 | 18079050 | 0.353 | SR. Web Designer |

Observed cosine scores in the run span roughly **0.0–0.45**. Run the notebook to reproduce the full ranked match table.

## Key Takeaways
- Resume screening is framed as document similarity, not classification — TF-IDF + cosine ranks candidates without labeled relevance data.
- Robust PDF text extraction and domain-noise word removal matter a lot for clean vectors.
- Capping `max_features=800` keeps the TF-IDF space compact and comparable across resumes and job descriptions.
- Cosine scores are relatively low (≤0.45), reflecting sparse lexical overlap; semantic embeddings (e.g. SBERT) would likely improve ranking.

## How to Run
```bash
pip install pypdf nltk scikit-learn wordcloud matplotlib pandas numpy
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
jupyter notebook notebook.ipynb
```
Place the resume CSV/PDFs and job-description data where the notebook expects `/kaggle/input/...` (or adjust the paths).

## Credit
> Based on ["Resume Screening using NLP"](https://www.kaggle.com/code/suditipandey/resume-screening-using-nlp) by suditi pandey on Kaggle (55 votes).
> Datasets: [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), [Resume and Job Description](https://www.kaggle.com/datasets/pranavvenugo/resume-and-job-description). Adapted and documented for this portfolio.
