import os
import csv
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.cosine_similarity import search
from src.evaluation import evaluate_response
from src.prompt import RAG_prompt

dotenv.load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("❌ API key is missing. Please check your .env file.")
    exit()

llm = ChatOpenAI(model="gpt-5.5", temperature=0)

questions = [
    # Paper 1 — Attention Is All You Need
    {"paper": "Attention Is All You Need", "question": "What problem does the Transformer solve compared to RNNs?", "gold": "RNNs are sequential and preclude parallelization during training. The Transformer relies entirely on self-attention, enabling full parallelization."},
    {"paper": "Attention Is All You Need", "question": "What are the three types of attention used in the Transformer?", "gold": "Encoder-decoder attention, encoder self-attention, and decoder self-attention."},
    {"paper": "Attention Is All You Need", "question": "What is the formula for Scaled Dot-Product Attention?", "gold": "Attention(Q, K, V) = softmax(QK^T / √dk) · V"},
    {"paper": "Attention Is All You Need", "question": "Why do the authors scale by √dk in the attention formula?", "gold": "For large dk the dot products grow large, pushing softmax into regions with extremely small gradients. Dividing by √dk counteracts this."},
    {"paper": "Attention Is All You Need", "question": "How many attention heads do the Transformer models use?", "gold": "h = 8 parallel attention heads."},
    {"paper": "Attention Is All You Need", "question": "What is Multi-Head Attention and why is it used?", "gold": "It allows the model to jointly attend to information from different representation subspaces at different positions."},
    {"paper": "Attention Is All You Need", "question": "What positional encoding do the authors use?", "gold": "Sine and cosine functions of different frequencies, allowing the model to attend by relative positions."},
    {"paper": "Attention Is All You Need", "question": "What BLEU score did the Transformer achieve on WMT 2014 English-to-German?", "gold": "28.4 BLEU, outperforming all previously reported models including ensembles."},
    {"paper": "Attention Is All You Need", "question": "What BLEU score did the Transformer achieve on WMT 2014 English-to-French?", "gold": "41.0 BLEU, a new state of the art at less than 1/4 the training cost of the prior best model."},
    {"paper": "Attention Is All You Need", "question": "How many encoder and decoder layers does the base Transformer have?", "gold": "N = 6 identical layers in both encoder and decoder."},
    {"paper": "Attention Is All You Need", "question": "What is d_model in the base Transformer?", "gold": "dmodel = 512"},
    {"paper": "Attention Is All You Need", "question": "What regularization techniques are applied during training?", "gold": "Residual dropout (Pdrop = 0.1) and label smoothing (ε_ls = 0.1)."},
    {"paper": "Attention Is All You Need", "question": "How long did it take to train the big Transformer?", "gold": "3.5 days on eight GPUs."},
    {"paper": "Attention Is All You Need", "question": "What learning rate schedule is used?", "gold": "A custom schedule with linear warmup then decay proportional to the inverse square root of the step number, using warmup_steps."},
    {"paper": "Attention Is All You Need", "question": "What is the feed-forward network dimension d_ff in the base model?", "gold": "dff = 2048"},
    {"paper": "Attention Is All You Need", "question": "What task beyond translation is the Transformer tested on?", "gold": "English constituency parsing."},
    {"paper": "Attention Is All You Need", "question": "What is the maximum path length between two positions in a self-attention layer?", "gold": "O(1) — constant regardless of sequence length. (See Table 1.)"},
    {"paper": "Attention Is All You Need", "question": "What is the main advantage of self-attention over recurrence in sequential operations?", "gold": "Self-attention connects all positions with O(1) sequential operations vs O(n) for recurrence. (Table 1.)"},
    {"paper": "Attention Is All You Need", "question": "What hardware was used to train the Transformer?", "gold": "8 NVIDIA P100 GPUs."},
    {"paper": "Attention Is All You Need", "question": "How does the decoder produce a probability distribution over vocabulary?", "gold": "A softmax function converts the decoder output to predicted next-token probabilities."},

    # Paper 2 — BERT
    {"paper": "BERT", "question": "What does BERT stand for?", "gold": "Bidirectional Encoder Representations from Transformers."},
    {"paper": "BERT", "question": "What is the key difference between BERT and GPT?", "gold": "BERT is deeply bidirectional — it attends to both left and right context simultaneously. GPT is left-to-right only."},
    {"paper": "BERT", "question": "What are BERT's two pre-training objectives?", "gold": "Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)."},
    {"paper": "BERT", "question": "How does Masked Language Modeling (MLM) work?", "gold": "15% of all WordPiece tokens are masked at random; the model predicts the original token from context."},
    {"paper": "BERT", "question": "What is the Next Sentence Prediction (NSP) task?", "gold": "The model receives two sentences and predicts whether the second follows the first in the original document (50% positive, 50% random)."},
    {"paper": "BERT", "question": "What are the two model sizes in BERT?", "gold": "BERT-Base (L=12, H=768, A=12, 110M params) and BERT-Large (L=24, H=1024, A=16, 340M params)."},
    {"paper": "BERT", "question": "What pre-training corpus does BERT use?", "gold": "BooksCorpus (800M words) and English Wikipedia (2,500M words)."},
    {"paper": "BERT", "question": "What is the [CLS] token used for?", "gold": "It is prepended to every input; its final hidden state is used as the aggregate sequence representation for classification tasks."},
    {"paper": "BERT", "question": "What GLUE score does BERT-Large achieve?", "gold": "80.5%, a 7.7% absolute improvement over the prior state of the art."},
    {"paper": "BERT", "question": "What F1 score does BERT-Large achieve on SQuAD v1.1?", "gold": "93.2 F1, surpassing human-level performance."},
    {"paper": "BERT", "question": "What is fine-tuning in the context of BERT?", "gold": "Starting from pre-trained BERT parameters and training end-to-end on a labeled downstream task with an added output layer."},
    {"paper": "BERT", "question": "What is the difference between feature-based and fine-tuning approaches?", "gold": "Fine-tuning adjusts all pre-trained parameters; feature-based uses fixed pre-trained representations as features in a task-specific architecture."},
    {"paper": "BERT", "question": "What WordPiece vocabulary size does BERT use?", "gold": "30,000 token vocabulary."},
    {"paper": "BERT", "question": "How long did it take to pre-train BERT?", "gold": "4 days to complete pre-training."},
    {"paper": "BERT", "question": "What ablation study is done on pre-training objectives?", "gold": "Models are compared with and without NSP (No NSP) and with a left-to-right LM instead of MLM, showing both objectives contribute."},
    {"paper": "BERT", "question": "How does BERT handle the mismatch between [MASK] at pre-training and real tokens at fine-tuning?", "gold": "Of the 15% selected tokens: 80% replaced with [MASK], 10% random token, 10% unchanged."},
    {"paper": "BERT", "question": "What is the maximum sequence length BERT supports?", "gold": "512 tokens."},
    {"paper": "BERT", "question": "What F1 does BERT-Large achieve on CoNLL-2003 NER?", "gold": "92.8 F1."},
    {"paper": "BERT", "question": "What learning rates are searched for fine-tuning?", "gold": "5e-5, 4e-5, 3e-5, and 2e-5; best is chosen on the Dev set."},
    {"paper": "BERT", "question": "What linguistic phenomena do BERT's attention heads learn?", "gold": "Different heads capture different linguistic properties (syntactic relations, sentence boundaries, etc.)."},

    # Paper 3 — RAG
    {"paper": "RAG", "question": "What is the core idea of RAG?", "gold": "RAG combines a pre-trained seq2seq model (parametric memory) with a dense vector index of Wikipedia (non-parametric memory) to generate grounded answers."},
    {"paper": "RAG", "question": "What retriever does RAG use?", "gold": "Dense Passage Retriever (DPR), which uses BERT-based encoders for queries and documents and retrieves top-K by maximum inner product search."},
    {"paper": "RAG", "question": "What are the two RAG formulations?", "gold": "RAG-Sequence (same retrieved document for the full answer) and RAG-Token (different documents per generated token)."},
    {"paper": "RAG", "question": "What document corpus does RAG use?", "gold": "Wikipedia split into disjoint 100-word chunks, totalling 21M documents, indexed with FAISS."},
    {"paper": "RAG", "question": "How many documents does RAG retrieve per query?", "gold": "Top-K documents; K=5 is used in most experiments."},
    {"paper": "RAG", "question": "What generative model does RAG use?", "gold": "BART-large."},
    {"paper": "RAG", "question": "What Exact Match score does RAG achieve on Natural Questions?", "gold": "44.5 EM (RAG-Sequence)."},
    {"paper": "RAG", "question": "Can RAG's knowledge be updated without retraining?", "gold": "Yes — the non-parametric memory can be replaced to update the models' knowledge without retraining the model parameters."},
    {"paper": "RAG", "question": "What is 'Closed-Book QA' and how does RAG compare?", "gold": "Closed-Book QA relies entirely on model parameters with no retrieval. RAG outperforms it by grounding answers in retrieved documents."},
    {"paper": "RAG", "question": "What open-domain QA datasets are used for evaluation?", "gold": "Natural Questions, WebQuestions, CuratedTrec, and TriviaQA."},
    {"paper": "RAG", "question": "What is hallucination and why is it a problem?", "gold": "Parametric-only models may 'hallucinate' plausible but incorrect facts. RAG mitigates this by conditioning on retrieved text."},
    {"paper": "RAG", "question": "What vector search library is used to index the passages?", "gold": "FAISS (Facebook AI Similarity Search) with a Hierarchical Navigable Small World index."},
    {"paper": "RAG", "question": "What model produces document embeddings in DPR?", "gold": "A BERT-base document encoder."},
    {"paper": "RAG", "question": "How does RAG compare to BART in output diversity?", "gold": "RAG generates more diverse outputs — a higher ratio of distinct n-grams to total n-grams compared to BART."},
    {"paper": "RAG", "question": "What is the FEVER task and how does RAG perform?", "gold": "FEVER is a fact verification task. RAG achieves competitive results with pipelines using full Wikipedia retrieval."},
    {"paper": "RAG", "question": "What metric is used to evaluate open-domain QA?", "gold": "Exact Match (EM) — whether the predicted string exactly matches the gold answer."},
    {"paper": "RAG", "question": "Can RAG's knowledge index be updated without retraining?", "gold": "Yes — the non-parametric memory can be replaced to update the models' knowledge without retraining."},
    {"paper": "RAG", "question": "What happens when the correct answer is not in any retrieved document?", "gold": "The model achieves 11.8% accuracy in those cases — it can sometimes answer from parametric memory alone."},
    {"paper": "RAG", "question": "What TriviaQA score does RAG-Sequence achieve?", "gold": "56.8 EM on open-domain TriviaQA."},
]


def run_rag_pipeline(question: str, k: int = 2) -> str:
    """Mirrors exactly what app.py does — search, build context, call LLM."""
    top_hits = search(question, k=k)
    if not top_hits:
        return ""

    context_block = ""
    for match in top_hits:
        context_block += f"--- START CHUNK FROM {match['pdf_name']} (Page {match['page']}) (Line {match['lines']}) ---\n"
        context_block += f"{match['text']}\n"
        context_block += "--- END CHUNK ---\n\n"

    system = RAG_prompt.format(context=context_block, question=question)
    message = [SystemMessage(content=system), HumanMessage(content="Please answer the question")]
    rag_chain = llm | StrOutputParser()
    return rag_chain.invoke(message)


if __name__ == "__main__":
    current_paper = None
    total = len(questions)
    errors = 0

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bert_f1_scores = []

    CSV_OUT = "evaluation_results.csv"
    csv_rows = []

    for i, entry in enumerate(questions, start=1):
        if entry["paper"] != current_paper:
            current_paper = entry["paper"]
            print("\n" + "=" * 80)
            print(f"  PAPER: {current_paper}")
            print("=" * 80)

        print(f"\n[Q{i}] {entry['question']}")

        # Run the real RAG pipeline
        try:
            generated_answer = run_rag_pipeline(entry["question"])
        except Exception as e:
            print(f"  PIPELINE ERROR: {e}")
            errors += 1
            csv_rows.append({
                "q_num": i, "paper": entry["paper"], "question": entry["question"],
                "gold": entry["gold"], "generated": "", "error": str(e),
                "rouge1": "", "rouge2": "", "rougeL": "",
                "bert_precision": "", "bert_recall": "", "bert_f1": ""
            })
            continue

        if not generated_answer:
            print("  ERROR: No answer generated (empty RAG result)")
            errors += 1
            csv_rows.append({
                "q_num": i, "paper": entry["paper"], "question": entry["question"],
                "gold": entry["gold"], "generated": "", "error": "Empty RAG result",
                "rouge1": "", "rouge2": "", "rougeL": "",
                "bert_precision": "", "bert_recall": "", "bert_f1": ""
            })
            continue

        print(f"  Generated: {generated_answer}")
        print(f"  Gold:      {entry['gold']}")

        result = evaluate_response(entry["question"], generated_answer)

        if "error" in result:
            print(f"  NOTE: {result['error']} — skipping metric display")
            errors += 1
            csv_rows.append({
                "q_num": i, "paper": entry["paper"], "question": entry["question"],
                "gold": entry["gold"], "generated": generated_answer, "error": result["error"],
                "rouge1": "", "rouge2": "", "rougeL": "",
                "bert_precision": "", "bert_recall": "", "bert_f1": ""
            })
        else:
            rouge1_scores.append(result["rouge1"])
            rouge2_scores.append(result["rouge2"])
            rougeL_scores.append(result["rougeL"])
            bert_f1_scores.append(result["bert_f1"])

            print(f"  ROUGE-1:        {result['rouge1']:.4f}")
            print(f"  ROUGE-2:        {result['rouge2']:.4f}")
            print(f"  ROUGE-L:        {result['rougeL']:.4f}")
            print(f"  BERT Precision: {result['bert_precision']:.4f}")
            print(f"  BERT Recall:    {result['bert_recall']:.4f}")
            print(f"  BERT F1:        {result['bert_f1']:.4f}")

            csv_rows.append({
                "q_num": i, "paper": entry["paper"], "question": entry["question"],
                "gold": entry["gold"], "generated": generated_answer, "error": "",
                "rouge1": round(result["rouge1"], 4),
                "rouge2": round(result["rouge2"], 4),
                "rougeL": round(result["rougeL"], 4),
                "bert_precision": round(result["bert_precision"], 4),
                "bert_recall": round(result["bert_recall"], 4),
                "bert_f1": round(result["bert_f1"], 4),
            })

        print("-" * 80)

    # Summary
    n = len(rouge1_scores)
    print("\n" + "=" * 80)
    print(f"  SUMMARY — {n}/{total} questions evaluated ({errors} errors/skipped)")
    print("=" * 80)
    if n > 0:
        print(f"  Avg ROUGE-1:  {sum(rouge1_scores)/n:.4f}")
        print(f"  Avg ROUGE-2:  {sum(rouge2_scores)/n:.4f}")
        print(f"  Avg ROUGE-L:  {sum(rougeL_scores)/n:.4f}")
        print(f"  Avg BERT F1:  {sum(bert_f1_scores)/n:.4f}")
    print("=" * 80)

    # Save to CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["q_num", "paper", "question", "gold", "generated", "error",
                      "rouge1", "rouge2", "rougeL", "bert_precision", "bert_recall", "bert_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✅ Results saved to {CSV_OUT}")