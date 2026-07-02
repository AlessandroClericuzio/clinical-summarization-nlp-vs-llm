import os
import time
import argparse
import logging
import pandas as pd

from src.NLPPipeline import ClassicalPipeline
from src.LLMPipeline import LLMPipeline
from src.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME      = "ccdv/pubmed-summarization"
DEFAULT_SPLIT             = "train"
DEFAULT_SAMPLE_SIZE       = 500
DEFAULT_OUTPUT_DIR        = "results/"
DEFAULT_LLM_MODEL         = "llama3.1:8b"
DEFAULT_PROMPTING         = "few-shot"
DEFAULT_TEMPERATURE       = 0.1
DEFAULT_MAX_NEW_TOKENS    = 2048
DEFAULT_POSITION_BIAS     = 0.25


def load_dataset_from_csv(csv_path: str, sample_size: int | None = None) -> pd.DataFrame:
    # Load and sample the processed validation or training dataset
    logger.info(f"Loading cleaned dataset from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File {csv_path} not found. "
            "Ensure you ran the exploration.ipynb notebook first "
            "to generate the cleaned dataset."
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)} columns")

    required = {"article", "abstract"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"Selected sample: {len(df)} samples")

    return df


def save_results(df_results: pd.DataFrame, output_dir: str, filename: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    df_results.to_csv(out_path, index=False)
    logger.info(f"Results saved to: {out_path}")
    return out_path


def print_summary_table(metrics: dict) -> None:
    print("\nFINAL RESULTS - PIPELINE COMPARISON")
    for pipeline_name, m in metrics.items():
        print(f"\nPipeline: {pipeline_name}")
        print(f"   {'Metric':<30} {'Value':>10}")
        for key, val in m.items():
            if key.endswith("_per_sample"):
                continue
            if isinstance(val, float):
                print(f"   {key:<30} {val:>10.4f}")
            else:
                print(f"   {key:<30} {str(val):>10}")


def run_pipeline_with_timing(
    pipeline, texts: list[str], label: str
) -> tuple[list[str], float]:
    logger.info(f"Starting {label} on {len(texts)} samples...")
    t_start = time.perf_counter()
    outputs = pipeline.run_batch(texts)
    elapsed = time.perf_counter() - t_start
    logger.info(
        f"{label} completed in {elapsed:.2f}s "
        f"({elapsed / len(texts) * 1000:.1f} ms/sample)"
    )
    return outputs, elapsed


def main(args: argparse.Namespace) -> None:
    # Load input dataset
    df = load_dataset_from_csv(
        csv_path="data/pubmed_cleaned.csv",
        sample_size=args.sample_size,
    )
    articles  = df["article"].tolist()
    abstracts = df["abstract"].tolist()

    # Initialize execution pipelines
    logger.info(
        f"Initializing Pipeline A (Classical - Extractive) | "
        f"n_sentences=5 | position_bias={args.position_bias_weight}"
    )
    pipeline_a = ClassicalPipeline(
        n_summary_sentences=5,
        position_bias_weight=args.position_bias_weight,
    )

    logger.info(
        f"Initializing Pipeline B (LLM - {args.prompting} prompting) | "
        f"model: {args.llm_model}"
    )
    pipeline_b = LLMPipeline(
        model_name=args.llm_model,
        prompting_strategy=args.prompting,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Execute summary generation
    summaries_a, time_a = run_pipeline_with_timing(pipeline_a, articles, "Pipeline A")
    summaries_b, time_b = run_pipeline_with_timing(pipeline_b, articles, "Pipeline B")

    # Evaluate results using reference metrics
    logger.info("Computing evaluation metrics...")
    evaluator = Evaluator(original_texts=articles)

    metrics_a = evaluator.evaluate(
        predictions=summaries_a,
        references=abstracts,
        pipeline_label="Pipeline_A_Classical",
        exec_time_seconds=time_a,
        compute_hallucinations=False,
    )

    metrics_b = evaluator.evaluate(
        predictions=summaries_b,
        references=abstracts,
        pipeline_label="Pipeline_B_LLM",
        exec_time_seconds=time_b,
        compute_hallucinations=True,
    )

    # Output and export results
    print_summary_table({
        "Pipeline A - Extractive (TextRank + Position Bias)": metrics_a,
        f"Pipeline B - LLM ({args.prompting})": metrics_b,
    })

    df_out = df.copy()
    df_out["summary_classical"] = summaries_a
    df_out["summary_llm"] = summaries_b
    
    predictions_filename = f"predictions_{args.prompting}.csv"
    metrics_filename = f"metrics_summary_{args.prompting}.csv"
    
    save_results(df_out, args.output_dir, predictions_filename)

    metrics_rows = []
    for pipeline_name, m in [("Classical", metrics_a), ("LLM", metrics_b)]:
        row = {"pipeline": pipeline_name}
        row.update({k: v for k, v in m.items() if not k.endswith("_per_sample")})
        metrics_rows.append(row)
    df_metrics = pd.DataFrame(metrics_rows)
    save_results(df_metrics, args.output_dir, metrics_filename)

    logger.info("Execution completed.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clinical Summarization - PubMed dataset"
    )
    parser.add_argument("--dataset_name",          type=str,   default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split",                 type=str,   default=DEFAULT_SPLIT)
    parser.add_argument("--sample_size",           type=int,   default=DEFAULT_SAMPLE_SIZE,
                        help="Number of samples to use (None = all)")
    parser.add_argument("--output_dir",            type=str,   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm_model",             type=str,   default=DEFAULT_LLM_MODEL)
    parser.add_argument("--prompting",             type=str,   default=DEFAULT_PROMPTING,
                        choices=["zero-shot", "one-shot", "few-shot", "cot"])
    parser.add_argument("--temperature",           type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_new_tokens",        type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--position_bias_weight",  type=float, default=DEFAULT_POSITION_BIAS,
                        help="Pipeline A position bias toward the second half of the text [0.0, 1.0]")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)