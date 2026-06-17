"""
main.py — Orchestratore principale
Progetto NLP: Clinical Summarization — NLP Tradizionale vs LLM
Dataset: PubMed Summarization (ccdv/pubmed-summarization)
"""

import os
import time
import argparse
import logging
import pandas as pd
from datasets import load_dataset

from src.NLPPipeline import ClassicalPipeline
from src.LLMPipeline import LLMPipeline
from src.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default
DEFAULT_DATASET_NAME = "ccdv/pubmed-summarization"
DEFAULT_SPLIT = "train"               # useremo il train set per il campione
DEFAULT_SAMPLE_SIZE = 500             # 500 esempi per tempi ragionevoli
DEFAULT_OUTPUT_DIR = "results/"
DEFAULT_LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"   # o un modello 7B con 32k context
DEFAULT_PROMPTING = "few-shot"        # zero-shot, few-shot, cot
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_NEW_TOKENS = 256          # aumentato per eguagliare la lunghezza degli abstract

def load_dataset_from_csv(csv_path: str, sample_size: int | None = None) -> pd.DataFrame:
    """
    Carica il dataset PubMed già pulito dal file CSV generato dal notebook di esplorazione.
    
    Args:
        csv_path: Percorso del file CSV (es. 'data/pubmed_cleaned.csv')
        sample_size: Numero di esempi da usare (None = tutto il dataset)
    
    Returns:
        DataFrame con le colonne 'article' e 'abstract'
    """
    logger.info(f"Caricamento dataset pulito da: {csv_path}")
    
    # Verifica che il file esista
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File {csv_path} non trovato. "
            "Assicurati di aver eseguito prima il notebook exploration.ipynb "
            "per generare il dataset pulito."
        )
    
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset caricato: {len(df)} esempi, {len(df.columns)} colonne")
    
    # Verifica colonne richieste
    required = {"article", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {missing}. Trovate: {list(df.columns)}")
    
    # Campionamento
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Campione selezionato: {len(df)} esempi")
    
    return df

def save_results(df_results: pd.DataFrame, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df_results.to_csv(out_path, index=False)
    logger.info(f"Risultati salvati in: {out_path}")
    return out_path


def print_summary_table(metrics: dict) -> None:
    separator = "─" * 55
    print(f"\n{'═' * 55}")
    print(f"  RISULTATI FINALI — CONFRONTO PIPELINE")
    print(f"{'═' * 55}")
    for pipeline_name, m in metrics.items():
        print(f"\n  📌 {pipeline_name}")
        print(separator)
        print(f"  {'Metrica':<30} {'Valore':>10}")
        print(separator)
        for key, val in m.items():
            if key.endswith("_per_sample"):
                continue
            if isinstance(val, float):
                print(f"  {key:<30} {val:>10.4f}")
            else:
                print(f"  {key:<30} {str(val):>10}")
    print(f"\n{'═' * 55}\n")


def run_pipeline_with_timing(pipeline, texts: list[str], label: str) -> tuple[list[str], float]:
    logger.info(f"Avvio {label} su {len(texts)} esempi...")
    t_start = time.perf_counter()
    outputs = pipeline.run_batch(texts)
    elapsed = time.perf_counter() - t_start
    logger.info(f"{label} completata in {elapsed:.2f}s "
                f"({elapsed / len(texts) * 1000:.1f} ms/esempio)")
    return outputs, elapsed


def main(args: argparse.Namespace) -> None:
    # 1. Caricamento dataset
    # Nel main.py, sostituisci il caricamento con:
    df = load_dataset_from_csv(
        csv_path="data/pubmed_cleaned.csv",
        sample_size=args.sample_size  # opzionale
    )
    articles   = df["article"].tolist()
    abstracts  = df["abstract"].tolist()

    # 2. Inizializzazione pipeline
    logger.info("Inizializzazione Pipeline A (Classica — Estrattiva)...")
    pipeline_a = ClassicalPipeline(n_summary_sentences=3)   # 3 frasi per un riassunto di ~60 parole

    logger.info(f"Inizializzazione Pipeline B (LLM — {args.prompting} prompting)...")
    pipeline_b = LLMPipeline(
        model_name=args.llm_model,
        prompting_strategy=args.prompting,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        use_4bit=True,
    )

    # 3. Esecuzione
    summaries_a, time_a = run_pipeline_with_timing(pipeline_a, articles, "Pipeline A")
    summaries_b, time_b = run_pipeline_with_timing(pipeline_b, articles, "Pipeline B")

    # 4. Valutazione
    logger.info("Calcolo metriche di valutazione...")
    evaluator = Evaluator(original_texts=articles)

    metrics_a = evaluator.evaluate(
        predictions=summaries_a,
        references=abstracts,
        pipeline_label="Pipeline_A_Classical",
        exec_time_seconds=time_a,
        compute_hallucinations=False,   # per pipeline A non calcoliamo allucinazioni (non genera nuovo testo)
    )

    metrics_b = evaluator.evaluate(
        predictions=summaries_b,
        references=abstracts,
        pipeline_label="Pipeline_B_LLM",
        exec_time_seconds=time_b,
        compute_hallucinations=True,
    )

    # 5. Stampa e salvataggio
    print_summary_table({
        "Pipeline A — Estrattiva (TextRank)": metrics_a,
        f"Pipeline B — LLM ({args.prompting})": metrics_b,
    })

    df_out = df.copy()
    df_out["summary_classical"] = summaries_a
    df_out["summary_llm"] = summaries_b
    save_results(df_out, args.output_dir, "predictions.csv")

    # Metriche aggregate
    metrics_rows = []
    for pipeline_name, m in [("Classical", metrics_a), ("LLM", metrics_b)]:
        row = {"pipeline": pipeline_name}
        row.update({k: v for k, v in m.items() if not k.endswith("_per_sample")})
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)
    save_results(df_metrics, args.output_dir, "metrics_summary.csv")

    logger.info("✅ Esecuzione completata.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical Summarization — PubMed dataset"
    )
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE,
                        help="Numero di esempi da usare (None = tutto)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--prompting", type=str, default=DEFAULT_PROMPTING,
                        choices=["zero-shot", "few-shot", "cot"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)