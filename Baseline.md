# PDF QA Eval Dataset — AI/ML Papers
# Page and line numbers verified by parsing the uploaded PDFs

Papers used:
- **Paper 1**: Attention Is All You Need — `1706.03762v7.pdf`
- **Paper 2**: BERT — `1810.04805v2.pdf`
- **Paper 3**: RAG (Lewis et al., 2020) — `2005.11401v4.pdf`

> **Scoring guide:** For each question, mark ✅ if retrieval@5 surfaced the correct page, and ✅ if the generated answer matches the gold answer. Page numbers are 1-indexed. Line numbers are counted from the top of that page.

---

## Paper 1 — Attention Is All You Need (Vaswani et al., 2017)
**File:** `1706.03762v7.pdf`

| # | Question | Gold Answer | Page | Line |
|---|----------|-------------|------|------|
| 1 | What problem does the Transformer solve compared to RNNs? | RNNs are sequential and preclude parallelization during training. The Transformer relies entirely on self-attention, enabling full parallelization. | 2 | 10 |
| 2 | What are the three types of attention used in the Transformer? | Encoder-decoder attention, encoder self-attention, and decoder self-attention. | 5 | 20 |
| 3 | What is the formula for Scaled Dot-Product Attention? | Attention(Q, K, V) = softmax(QK^T / √dk) · V | 4 | 14 |
| 4 | Why do the authors scale by √dk in the attention formula? | For large dk the dot products grow large, pushing softmax into regions with extremely small gradients. Dividing by √dk counteracts this. | 4 | 27 |
| 5 | How many attention heads do the Transformer models use? | h = 8 parallel attention heads. | 5 | 15 |
| 6 | What is Multi-Head Attention and why is it used? | It allows the model to jointly attend to information from different representation subspaces at different positions. | 5 | 3 |
| 7 | What positional encoding do the authors use? | Sine and cosine functions of different frequencies, allowing the model to attend by relative positions. | 6 | 17 |
| 8 | What BLEU score did the Transformer achieve on WMT 2014 English-to-German? | 28.4 BLEU, outperforming all previously reported models including ensembles. | 1 | 36 |
| 9 | What BLEU score did the Transformer achieve on WMT 2014 English-to-French? | 41.0 BLEU, a new state of the art at less than 1/4 the training cost of the prior best model. | 8 | 30 |
| 10 | How many encoder and decoder layers does the base Transformer have? | N = 6 identical layers in both encoder and decoder. | 3 | 6 |
| 11 | What is d_model in the base Transformer? | dmodel = 512 | 3 | 12 |
| 12 | What regularization techniques are applied during training? | Residual dropout (Pdrop = 0.1) and label smoothing (ε_ls = 0.1). | 8 | 19 |
| 13 | How long did it take to train the big Transformer? | 3.5 days on eight GPUs. | 1 | 40 |
| 14 | What learning rate schedule is used? | A custom schedule with linear warmup then decay proportional to the inverse square root of the step number, using warmup_steps. | 7 | 39 |
| 15 | What is the feed-forward network dimension d_ff in the base model? | dff = 2048 | 5 | 42 |
| 16 | What task beyond translation is the Transformer tested on? | English constituency parsing. | 1 | 42 |
| 17 | What is the maximum path length between two positions in a self-attention layer? | O(1) — constant regardless of sequence length. (See Table 1.) | 6 | 1 |
| 18 | What is the main advantage of self-attention over recurrence in sequential operations? | Self-attention connects all positions with O(1) sequential operations vs O(n) for recurrence. (Table 1.) | 6 | 1 |
| 19 | What hardware was used to train the Transformer? | 8 NVIDIA P100 GPUs. | 7 | 30 |
| 20 | How does the decoder produce a probability distribution over vocabulary? | A softmax function converts the decoder output to predicted next-token probabilities. | 5 | 46 |

---

## Paper 2 — BERT (Devlin et al., 2018)
**File:** `1810.04805v2.pdf`

| # | Question | Gold Answer | Page | Line |
|---|----------|-------------|------|------|
| 1 | What does BERT stand for? | Bidirectional Encoder Representations from Transformers. | 1 | 1 |
| 2 | What is the key difference between BERT and GPT? | BERT is deeply bidirectional — it attends to both left and right context simultaneously. GPT is left-to-right only. | 2 | 18 |
| 3 | What are BERT's two pre-training objectives? | Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). | 4 | 38 |
| 4 | How does Masked Language Modeling (MLM) work? | 15% of all WordPiece tokens are masked at random; the model predicts the original token from context. | 4 | 60 |
| 5 | What is the Next Sentence Prediction (NSP) task? | The model receives two sentences and predicts whether the second follows the first in the original document (50% positive, 50% random). | 2 | 7 |
| 6 | What are the two model sizes in BERT? | BERT-Base (L=12, H=768, A=12, 110M params) and BERT-Large (L=24, H=1024, A=16, 340M params). | 3 | 80 |
| 7 | What pre-training corpus does BERT use? | BooksCorpus (800M words) and English Wikipedia (2,500M words). | 5 | 20 |
| 8 | What is the [CLS] token used for? | It is prepended to every input; its final hidden state is used as the aggregate sequence representation for classification tasks. | 3 | 2 |
| 9 | What GLUE score does BERT-Large achieve? | 80.5%, a 7.7% absolute improvement over the prior state of the art. | 1 | 25 |
| 10 | What F1 score does BERT-Large achieve on SQuAD v1.1? | 93.2 F1, surpassing human-level performance. | 1 | 29 |
| 11 | What is fine-tuning in the context of BERT? | Starting from pre-trained BERT parameters and training end-to-end on a labeled downstream task with an added output layer. | 2 | 82 |
| 12 | What is the difference between feature-based and fine-tuning approaches? | Fine-tuning adjusts all pre-trained parameters; feature-based uses fixed pre-trained representations as features in a task-specific architecture. | 1 | 49 |
| 13 | What WordPiece vocabulary size does BERT use? | 30,000 token vocabulary. | 4 | 12 |
| 14 | How long did it take to pre-train BERT? | 4 days to complete pre-training. | 13 | 79 |
| 15 | What ablation study is done on pre-training objectives? | Models are compared with and without NSP (No NSP) and with a left-to-right LM instead of MLM, showing both objectives contribute. | 8 | 4 |
| 16 | How does BERT handle the mismatch between [MASK] at pre-training and real tokens at fine-tuning? | Of the 15% selected tokens: 80% replaced with [MASK], 10% random token, 10% unchanged. | 12 | 72 |
| 17 | What is the maximum sequence length BERT supports? | 512 tokens. | 13 | 57 |
| 18 | What F1 does BERT-Large achieve on CoNLL-2003 NER? | 92.8 F1. | 9 | 55 |
| 19 | What learning rates are searched for fine-tuning? | 5e-5, 4e-5, 3e-5, and 2e-5; best is chosen on the Dev set. | 6 | 16 |
| 20 | What linguistic phenomena do BERT's attention heads learn? | Different heads capture different linguistic properties (syntactic relations, sentence boundaries, etc.). | 4 | 7 |

---

## Paper 3 — RAG (Lewis et al., 2020)
**File:** `2005.11401v4.pdf`

| # | Question | Gold Answer | Page | Line |
|---|----------|-------------|------|------|
| 1 | What is the core idea of RAG? | RAG combines a pre-trained seq2seq model (parametric memory) with a dense vector index of Wikipedia (non-parametric memory) to generate grounded answers. | 1 | 20 |
| 2 | What retriever does RAG use? | Dense Passage Retriever (DPR), which uses BERT-based encoders for queries and documents and retrieves top-K by maximum inner product search. | 2 | 54 |
| 3 | What are the two RAG formulations? | RAG-Sequence (same retrieved document for the full answer) and RAG-Token (different documents per generated token). | 3 | 5 |
| 4 | What document corpus does RAG use? | Wikipedia split into disjoint 100-word chunks, totalling 21M documents, indexed with FAISS. | 4 | 29 |
| 5 | How many documents does RAG retrieve per query? | Top-K documents; K=5 is used in most experiments. | 2 | 45 |
| 6 | What generative model does RAG use? | BART-large. | 2 | 55 |
| 7 | What Exact Match score does RAG achieve on Natural Questions? | 44.5 EM (RAG-Sequence). | 6 | 15 |
| 8 | Can RAG's knowledge be updated without retraining? | Yes — the non-parametric memory can be replaced to update the models' knowledge without retraining the model parameters. | 2 | 76 |
| 9 | What is "Closed-Book QA" and how does RAG compare? | Closed-Book QA relies entirely on model parameters with no retrieval. RAG outperforms it by grounding answers in retrieved documents. | 4 | 40 |
| 10 | What open-domain QA datasets are used for evaluation? | Natural Questions, WebQuestions, CuratedTrec, and TriviaQA. | 2 | 69 |
| 11 | What is hallucination and why is it a problem? | Parametric-only models may "hallucinate" plausible but incorrect facts. RAG mitigates this by conditioning on retrieved text. | 1 | 35 |
| 12 | What vector search library is used to index the passages? | FAISS (Facebook AI Similarity Search) with a Hierarchical Navigable Small World index. | 4 | 31 |
| 13 | What model produces document embeddings in DPR? | A BERT-base document encoder. | 3 | 45 |
| 14 | How does RAG compare to BART in output diversity? | RAG generates more diverse outputs — a higher ratio of distinct n-grams to total n-grams compared to BART. | 7 | 64 |
| 15 | What is the FEVER task and how does RAG perform? | FEVER is a fact verification task. RAG achieves competitive results with pipelines using full Wikipedia retrieval. | 2 | 74 |
| 16 | What is the embedding dimension used by DPR? | 768 dimensions (BERT-base hidden size). *(Note: not stated explicitly in this paper; see DPR paper arXiv:2004.04906.)* | N/A | N/A |
| 17 | What metric is used to evaluate open-domain QA? | Exact Match (EM) — whether the predicted string exactly matches the gold answer. | 4 | 45 |
| 18 | Can RAG's knowledge index be updated without retraining? | Yes — the non-parametric memory can be replaced to update the models' knowledge without retraining. | 2 | 76 |
| 19 | What happens when the correct answer is not in any retrieved document? | The model achieves 11.8% accuracy in those cases — it can sometimes answer from parametric memory alone. | 6 | 26 |
| 20 | What TriviaQA score does RAG-Sequence achieve? | 56.8 EM on open-domain TriviaQA. | 6 | 15 |

---

## Scoring template

| Paper | Q | Retrieval@5 ✅/❌ | Answer correct ✅/❌ | Citation correct ✅/❌ |
|-------|---|------------------|---------------------|----------------------|
| Attention | 1–20 | | | |
| BERT | 1–20 | | | |
| RAG | 1–20 | | | |

**Retrieval@5**: Did the correct page appear in the top 5 retrieved chunks?  
**Answer correct**: Does the generated answer match the gold answer?  
**Citation correct**: Did the system cite the correct page number?
