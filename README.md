# Sequence Models for NLP with PyTorch — RNN / GRU / LSTM and GPT-2


| Notebook | Task |
|---|---|
| `Part1_Arabic_Relevance_RNN_GRU_LSTM.ipynb` | Score the relevance (0–10) of scraped Arabic texts with **RNN, BiRNN, GRU and LSTM** |
| `Part2_GPT2_Finetuning_Text_Generation.ipynb` | **Fine-tune GPT-2** on a custom dataset and generate a paragraph from a given sentence |

Both notebooks run on Google Colab / Kaggle (dependencies are installed in their first cell).

---

## Part 1 — Relevance scoring of Arabic texts

**Data.** Arabic pages about *Artificial Intelligence* scraped with `requests` + BeautifulSoup from Arabic Wikipedia, Al Jazeera and Sky News Arabia (AI pages plus unrelated pages, so the full 0–10 range exists): **1,743 text chunks from 211 pages**. The score is computed automatically (60 % keyword-lexicon density + 40 % TF-IDF similarity to a topic profile, scaled to 0–10), not hand-labelled.

**Preprocessing.** Arabic normalisation (diacritics, tatweel, alef/ya/ta-marbuta variants), tokenisation, stop-word removal (NLTK + custom list), ISRI stemming, `qalsadi` lemmatisation, discretisation of the score into low / medium / high. Split 70/15/15, grouped by page so that no article is shared between train and test.

**Models.** Embedding → RNN / BiRNN / GRU / LSTM → pooling → MLP, trained as regressors (MSE, AdamW, early stopping). 8 random hyper-parameter configurations per architecture, selected on the validation set. For BLEU, each architecture is also trained as a small word-level language model.

**Results (193 test texts):**

| Model | RMSE ↓ | R² ↑ | Macro-F1 ↑ | LM perplexity ↓ | BLEU-1 |
|---|---:|---:|---:|---:|---:|
| RNN | 0.918 | 0.722 | 0.649 | 597.3 | 0.103 |
| BiRNN | 1.055 | 0.633 | 0.639 | 600.1 | 0.111 |
| GRU | 0.841 | 0.766 | **0.726** | 593.1 | 0.105 |
| LSTM | **0.797** | **0.790** | 0.713 | **552.4** | 0.101 |

- **GRU and LSTM are the best and are close to each other** (one seed, small test set). The LSTM leads on RMSE, R² and perplexity, the GRU on macro-F1.
- With last-hidden-state pooling the vanilla RNN breaks down (validation RMSE 2.16, versus 1.14 with mean pooling); the GRU is much less affected (0.91 vs 0.79). This is consistent with vanishing gradients on ~100-token sequences.
- The bidirectional RNN brings no gain: relevance depends on keywords, not on right-hand context.
- BLEU is very low for all models (BLEU-4 < 0.001) and the differences are not meaningful; perplexity is the more informative language-model metric.

**Limits.** The score comes from a keyword lexicon, and the most frequent vocabulary words are the lexicon words, so a high R² partly measures how well the lexicon was re-learned rather than human-judged relevance. The class thresholds (1.0 and 2.0 out of 10) make "medium" very narrow, and 62 % of the test set is "low", so accuracy is flattering: rely on RMSE/R² and macro-F1. Euronews and i24NEWS returned no articles and are not used. One seed.

---

## Part 2 — Fine-tuning GPT-2 for text generation

**Dataset (built by the notebook).** Paragraphs about AI / machine learning from English Wikipedia through the MediaWiki API (GPT-2 is an English model), cleaned and cut at sentence boundaries. **35 of the 50 planned articles were collected** (15 returned nothing, probably API throttling, including GPT-2, Large language model, Natural language processing and Backpropagation), giving **1,535 paragraphs, ≈ 118k words**. Split by article: 1,364 train / 102 validation / 69 test paragraphs (test articles: *Chatbot, Data science, Fine-tuning (deep learning), Perplexity*).

**Training.** GPT-2 small (124M parameters, Hugging Face `transformers`, the successor of `pytorch-transformers`), AdamW (lr 5e-5) with warm-up and linear decay, batch 8 × 2 gradient accumulation, fp16 mixed precision, best checkpoint kept by validation perplexity. About 1.6 minutes on a Colab GPU. The best epoch was the first one (validation perplexity 34.75, then 34.88 and 34.96 while the training loss kept falling: mild over-fitting).

**Results (held-out articles; 52 prompts = first sentence of a test paragraph, reference = the rest):**

| | Pre-trained GPT-2 | Fine-tuned GPT-2 |
|---|---:|---:|
| Perplexity, validation / test | 40.9 / 38.7 | **34.7 / 32.8** (−15 %) |
| BLEU-1 / 2 / 3 / 4 | 0.194 / 0.064 / 0.013 / 0.003 | 0.187 / 0.064 / 0.025 / 0.011 |
| distinct-1 / distinct-2 | 0.226 / 0.668 | 0.270 / 0.722 |
| Ends the paragraph by itself | 2 % | **90 %** |
| Words per output | 91 | 54 |

> **Prompt:** *The transformer architecture has changed the way language models are built.*
> **Pre-trained:** "…This means that you can now use any data model … (using JSON, XML or Java). 1 2 3 4 5 6 7 …" (it ends up counting numbers)
> **Fine-tuned:** "It was introduced in 2008 by Mark Lippmann and Alan Cogger, who were at Stanford University …" (encyclopaedic tone, but invented facts: the Transformer dates from 2017)

- Fine-tuning lowered held-out perplexity by 15 % and taught the model the shape of an encyclopaedic paragraph: it now stops by itself (90 % vs 2 %), is shorter, and no longer drifts into blog or interview style.
- **No BLEU gain** can be claimed (BLEU-1 is unchanged; BLEU-3/4 are higher but tiny and measured on 52 prompts).
- Decoding matters (one prompt, qualitative): greedy repeats a sentence, temperature 1.3 gives gibberish, and beam search with `no_repeat_ngram_size=3` gave the most coherent text.

**Limits.** Small dataset and only four held-out articles, so the differences are noisy; one seed. BLEU uses one reference and forces at least 100 new tokens, which prevents the fine-tuned model from stopping where it naturally would. distinct-n rises partly because the outputs are shorter. The generated text is fluent but not factual (hallucinations in both models), and the demo prompts cover topics present in the training articles, so they do not measure generalisation.

---
