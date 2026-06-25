import os
import pandas as pd
from rouge_score import rouge_scorer

CSV_PATH = "pdf_qa_eval_dataset_final.csv"

def load_ground_truth(csv_path=CSV_PATH):
    """Loads the ground truth dataset containing questions and golden answers."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def evaluate_response(user_question: str, generated_answer: str) -> dict:
    """
    Compares the generated answer against the golden answer from the dataset.
    Returns a dictionary with ROUGE-1 and ROUGE-2 scores.
    """
    df = load_ground_truth()
    if df is None:
        print("Error ground truth not found")
        return {"error": "Could not load ground truth dataset."}

    question_series = pd.Series(df['question']).astype(str)

    # Compare Pandas series (.str) against the native Python string (no .str)
    matched_rows = df[question_series.str.strip().str.lower() == user_question.strip().lower()]

    if matched_rows.empty:
        return {"error": "Question not found in the ground truth dataset."}

    golden_answer = matched_rows.iloc[0]['gold_answer']

    # Calculate ROUGE-1 and ROUGE-2 Scores using rouge-score library exclusively
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_results = scorer.score(golden_answer, generated_answer)

    return {
        "rouge1": rouge_results['rouge1'].fmeasure,
        "rouge2": rouge_results['rouge2'].fmeasure,
        "rougeL": rouge_results['rougeL'].fmeasure,
        "golden_answer": golden_answer
    }