"""
main.py — Orchestratore principale
Progetto NLP: Clinical Summarization — NLP Tradizionale vs LLM

Corso: Natural Language Processing (Laurea Magistrale)
Docenti: Prof.ssa Genoveffa Tortora, Prof.ssa Loredana Caruccio

Flusso:
  1. Caricamento e preprocessing del dataset MeQSum
  2. Esecuzione parallela Pipeline A (Estrattiva) e Pipeline B (LLM)
  3. Calcolo metriche di valutazione (ROUGE, F1 NER, allucinazioni, tempi)
  4. Salvataggio risultati e report finale
"""

import os
import time
import argparse
import logging
import pandas as pd

from src.NLPPipeline import ClassicalPipeline
from src.LLMPipeline import LLMPipeline
from src.evaluation import Evaluator

# ─────────────────────────────────────────────
# Configurazione logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Costanti di default
# ─────────────────────────────────────────────
DEFAULT_DATA_PATH   = "data/MeQSum_ACL2019_BenAbacha_Demner-Fushman.xlsx - QS.csv"
DEFAULT_OUTPUT_DIR  = "results/"
DEFAULT_SAMPLE_SIZE = 100          # None = usa tutto il dataset
DEFAULT_LLM_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_PROMPTING   = "few-shot"   # "zero-shot" | "few-shot" | "cot"


# ─────────────────────────────────────────────
# Utilità
# ─────────────────────────────────────────────
def load_dataset(path: str, sample_size: int | None) -> pd.DataFrame:
    """Carica il dataset MeQSum e restituisce un DataFrame pulito."""
    logger.info(f"Caricamento dataset da: {path}")

    ext = os.path.splitext(path)[-1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Normalizza nomi colonne (strip spazi, case-insensitive)
    df.columns = [c.strip() for c in df.columns]

    required = {"CHQ", "Summary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel dataset: {missing}. "
                         f"Colonne trovate: {list(df.columns)}")

    # Rimuovi righe con valori nulli nelle colonne chiave
    before = len(df)
    df = df.dropna(subset=["CHQ", "Summary"]).reset_index(drop=True)
    logger.info(f"Righe dopo pulizia NaN: {len(df)} (rimosse: {before - len(df)})")

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=42).reset_index(drop=True)
        logger.info(f"Campione selezionato: {len(df)} esempi")

    return df


def save_results(df_results: pd.DataFrame, output_dir: str, filename: str) -> str:
    """Salva il DataFrame dei risultati in CSV."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df_results.to_csv(out_path, index=False)
    logger.info(f"Risultati salvati in: {out_path}")
    return out_path


def print_summary_table(metrics: dict) -> None:
    """Stampa una tabella riassuntiva delle metriche a console."""
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
            if isinstance(val, float):
                print(f"  {key:<30} {val:>10.4f}")
            else:
                print(f"  {key:<30} {str(val):>10}")

    print(f"\n{'═' * 55}\n")


# ─────────────────────────────────────────────
# Pipeline runner con timing
# ─────────────────────────────────────────────
def run_pipeline_with_timing(pipeline, texts: list[str], label: str) -> tuple[list[str], float]:
    """Esegue una pipeline su tutti i testi e misura il tempo totale."""
    logger.info(f"Avvio {label} su {len(texts)} esempi...")
    t_start = time.perf_counter()
    outputs = pipeline.run_batch(texts)
    elapsed = time.perf_counter() - t_start
    logger.info(f"{label} completata in {elapsed:.2f}s "
                f"({elapsed / len(texts) * 1000:.1f} ms/esempio)")
    return outputs, elapsed


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:

    # 1. Caricamento dataset ──────────────────
    df = load_dataset(args.data_path, args.sample_size)
    chq_texts    = df["CHQ"].tolist()
    ground_truth = df["Summary"].tolist()

    # 2. Inizializzazione pipeline ────────────
    logger.info("Inizializzazione Pipeline A (Classica — Estrattiva)...")
    pipeline_a = ClassicalPipeline()

    logger.info(f"Inizializzazione Pipeline B (LLM — {args.prompting} prompting)...")
    pipeline_b = LLMPipeline(
        model_name=args.llm_model,
        prompting_strategy=args.prompting,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # 3. Esecuzione parallela (sequenziale su CPU/GPU singola) ──
    summaries_a, time_a = run_pipeline_with_timing(pipeline_a, chq_texts, "Pipeline A")
    summaries_b, time_b = run_pipeline_with_timing(pipeline_b, chq_texts, "Pipeline B")

    # 4. Valutazione ──────────────────────────
    logger.info("Calcolo metriche di valutazione...")
    evaluator = Evaluator(original_texts=chq_texts)

    metrics_a = evaluator.evaluate(
        predictions=summaries_a,
        references=ground_truth,
        pipeline_label="Pipeline_A_Classical",
        exec_time_seconds=time_a,
    )

    metrics_b = evaluator.evaluate(
        predictions=summaries_b,
        references=ground_truth,
        pipeline_label="Pipeline_B_LLM",
        exec_time_seconds=time_b,
    )

    # 5. Stampa riepilogo ─────────────────────
    print_summary_table({
        "Pipeline A — Estrattiva (TextRank + scispaCy)": metrics_a,
        f"Pipeline B — LLM ({args.prompting})":          metrics_b,
    })

    # 6. Salvataggio risultati ────────────────
    df_out = df.copy()
    df_out["summary_classical"]  = summaries_a
    df_out["summary_llm"]        = summaries_b

    # Aggiunge le metriche per riga se disponibili (ROUGE per campione)
    for metric_key in ["rouge1", "rouge2", "rougeL"]:
        if f"{metric_key}_per_sample" in metrics_a:
            df_out[f"classical_{metric_key}"] = metrics_a[f"{metric_key}_per_sample"]
        if f"{metric_key}_per_sample" in metrics_b:
            df_out[f"llm_{metric_key}"] = metrics_b[f"{metric_key}_per_sample"]

    save_results(df_out, args.output_dir, "predictions.csv")

    # Salva metriche aggregate in un file separato
    metrics_rows = []
    for pipeline_name, m in [("Classical", metrics_a), ("LLM", metrics_b)]:
        row = {"pipeline": pipeline_name}
        row.update({k: v for k, v in m.items() if not k.endswith("_per_sample")})
        metrics_rows.append(row)

    df_metrics = pd.DataFrame(metrics_rows)
    save_results(df_metrics, args.output_dir, "metrics_summary.csv")

    logger.info("✅ Esecuzione completata.")


# ─────────────────────────────────────────────
# Argomenti CLI
# ─────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical NLP Summarization — NLP Tradizionale vs LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path", type=str, default=DEFAULT_DATA_PATH,
        help="Percorso del dataset MeQSum (CSV o Excel)",
    )
    parser.add_argument(
        "--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Numero di esempi da usare (None = tutto il dataset)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Cartella di output per CSV e report",
    )
    parser.add_argument(
        "--llm_model", type=str, default=DEFAULT_LLM_MODEL,
        help="Nome o path del modello LLM (HuggingFace o locale)",
    )
    parser.add_argument(
        "--prompting", type=str, default=DEFAULT_PROMPTING,
        choices=["zero-shot", "few-shot", "cot"],
        help="Strategia di prompting per la Pipeline B",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Temperatura di decoding dell'LLM (bassa = meno allucinazioni)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="Numero massimo di token generati dall'LLM per ogni sintesi",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)