import logging
import re
import time
from typing import Optional, Callable

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score_fn

logger = logging.getLogger(__name__)

_HALLUCINATION_THRESHOLD = 0.15
_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]

def _normalize(text: str) -> str:
    # Lowercase text and remove extra punctuation
    return re.sub(r"[^\w\s]", " ", text.lower()).strip()


def _simple_stem(word: str) -> str:
    # Minimal heuristic suffix stripping for morphological evaluation
    for suffix in ("ations", "ation", "ings", "ing", "tions", "tion", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    # Computes standard ROUGE-1, ROUGE-2, and ROUGE-L F-measures
    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)
    per_sample = {rt: [] for rt in _ROUGE_TYPES}

    for pred, ref in zip(predictions, references):
        pred = pred.strip() if pred else ""
        ref  = ref.strip()  if ref  else ""

        if not pred or not ref:
            for rt in _ROUGE_TYPES:
                per_sample[rt].append(0.0)
            continue

        scores = scorer.score(ref, pred)
        for rt in _ROUGE_TYPES:
            per_sample[rt].append(scores[rt].fmeasure)

    results = {}
    for rt in _ROUGE_TYPES:
        results[rt] = float(np.mean(per_sample[rt]))
        results[f"{rt}_per_sample"] = per_sample[rt]

    return results


def compute_bertscore(predictions: list[str],
                      references: list[str],
                      model_type: str = "bert-base-uncased") -> dict:
    # Measures semantic similarity using contextual embeddings with baseline rescaling
    valid_preds, valid_refs, valid_indices = [], [], []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if pred and pred.strip() and ref and ref.strip():
            valid_preds.append(pred.strip())
            valid_refs.append(ref.strip())
            valid_indices.append(i)

    n = len(predictions)
    precision_scores = [0.0] * n
    recall_scores    = [0.0] * n
    f1_scores        = [0.0] * n

    if valid_preds:
        try:
            P, R, F1 = bert_score_fn(
                valid_preds,
                valid_refs,
                model_type=model_type,
                lang="en",
                verbose=False,
                device=None,
                rescale_with_baseline=True,
            )
            for list_pos, orig_idx in enumerate(valid_indices):
                precision_scores[orig_idx] = float(P[list_pos])
                recall_scores[orig_idx]    = float(R[list_pos])
                f1_scores[orig_idx]        = float(F1[list_pos])

        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}. Verify that 'bert-score' is installed.")
            import traceback
            logger.error(traceback.format_exc())

    return {
        "bertscore_precision":            float(np.mean(precision_scores)),
        "bertscore_recall":               float(np.mean(recall_scores)),
        "bertscore_f1":                   float(np.mean(f1_scores)),
        "bertscore_precision_per_sample": precision_scores,
        "bertscore_recall_per_sample":    recall_scores,
        "bertscore_f1_per_sample":        f1_scores,
    }


def compute_ner_f1(predicted_entities: list[set],
                   reference_entities: list[set]) -> dict:
    # Computes precision, recall, and F1 score for entity extraction against the reference source
    precision_scores, recall_scores, f1_scores = [], [], []

    for pred_set, ref_set in zip(predicted_entities, reference_entities):
        pred_set = {e.lower().strip() for e in pred_set if e.strip()}
        ref_set  = {e.lower().strip() for e in ref_set  if e.strip()}

        if not pred_set and not ref_set:
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            f1_scores.append(1.0)
            continue

        if not pred_set or not ref_set:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        tp = len(pred_set & ref_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set  - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "ner_precision":            float(np.mean(precision_scores)),
        "ner_recall":               float(np.mean(recall_scores)),
        "ner_f1":                   float(np.mean(f1_scores)),
        "ner_precision_per_sample": precision_scores,
        "ner_recall_per_sample":    recall_scores,
        "ner_f1_per_sample":        f1_scores,
    }


def _entity_in_context(entity: str, context: str,
                      context_tokens: Optional[set] = None,
                      vectorizer: Optional[TfidfVectorizer] = None) -> bool:
    # Verifies entity alignment using exact match, stem matching, or TF-IDF similarity fallback
    if not entity.strip() or not context.strip():
        return False

    entity_norm = _normalize(entity)
    context_norm = _normalize(context)

    if entity_norm in context_norm:
        return True

    entity_stems = {_simple_stem(t) for t in entity_norm.split() if len(t) > 2}
    if entity_stems and context_tokens is not None:
        if entity_stems.issubset(context_tokens):
            return True

    if vectorizer is not None:
        try:
            vec = vectorizer.transform([entity_norm])
            ctx_vec = vectorizer.transform([context_norm])
            sim = cosine_similarity(vec, ctx_vec)[0][0]
            return float(sim) >= _HALLUCINATION_THRESHOLD
        except Exception:
            pass

    return False


def compute_hallucination_rate(
    generated_summaries: list[str],
    original_texts: list[str],
    ner_fn: Callable,
) -> dict:
    # Calculates hallucination rate based on entities not anchored to the source text context
    rates        = []
    total_ents   = 0
    total_halluc = 0

    for summary, original in zip(generated_summaries, original_texts):
        summary  = summary.strip()  if isinstance(summary,  str) else ""
        original = original.strip() if isinstance(original, str) else ""

        if not summary:
            rates.append(0.0)
            continue

        entities = ner_fn(summary).get("entity_texts", set())

        if not entities:
            rates.append(0.0)
            continue

        context_norm   = _normalize(original)
        context_tokens = {_simple_stem(t) for t in context_norm.split() if len(t) > 2}

        vectorizer = None
        if original.strip():
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
                vectorizer.fit([context_norm])
            except Exception:
                vectorizer = None

        hallucinations = 0
        for entity in entities:
            anchored = _entity_in_context(
                entity, original,
                context_tokens=context_tokens,
                vectorizer=vectorizer,
                )
            if not anchored:
                hallucinations += 1

        rate = hallucinations / len(entities)
        rates.append(rate)
        total_ents   += len(entities)
        total_halluc += hallucinations

    return {
        "hallucination_rate":            float(np.mean(rates)) if rates else 0.0,
        "total_entities_generated":      total_ents,
        "total_hallucinations":          total_halluc,
        "hallucination_rate_per_sample": rates,
    }


def compute_timing_metrics(exec_time_seconds: float, n_samples: int) -> dict:
    # Computes computational efficiency metrics
    ms_per_example = (exec_time_seconds / n_samples * 1000) if n_samples > 0 else 0.0
    return {
        "exec_time_seconds": round(exec_time_seconds, 3),
        "ms_per_example":    round(ms_per_example, 2),
        "throughput_eps":    round(n_samples / exec_time_seconds, 2) if exec_time_seconds > 0 else 0.0,
    }


class Evaluator:
    # Unified evaluation framework comparing predictions against references and source texts

    def __init__(self, original_texts: list[str]):
        self.original_texts = original_texts

        try:
            from src.NLPPipeline import extract_medical_entities
            self._ner_fn = extract_medical_entities
            logger.info("Evaluator: scispaCy NER available")
        except ImportError:
            self._ner_fn = lambda text: {"entities": [], "entity_texts": set(), "count": 0}
            logger.warning("Evaluator: scispaCy not available, NER disabled")

        logger.info(f"Evaluator initialized on {len(original_texts)} samples")

    def _extract_entity_sets(self, texts: list[str]) -> list[set]:
        result = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                result.append(set())
                continue
            result.append(self._ner_fn(text).get("entity_texts", set()))
        return result

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        pipeline_label: str,
        exec_time_seconds: float,
        compute_hallucinations: bool = True,
    ) -> dict:
        n = len(predictions)
        logger.info(f"[{pipeline_label}] Computing metrics on {n} samples...")

        results = {"pipeline": pipeline_label, "n_samples": n}

        logger.info(f"[{pipeline_label}] Computing ROUGE...")
        rouge_metrics = compute_rouge(predictions, references)
        results.update(rouge_metrics)

        logger.info(f"[{pipeline_label}] Computing BERTScore...")
        bertscore_metrics = compute_bertscore(predictions, references)
        results.update(bertscore_metrics)

        logger.info(f"[{pipeline_label}] Computing NER F1 (vs original text)...")
        pred_entities = self._extract_entity_sets(predictions)
        orig_entities = self._extract_entity_sets(self.original_texts)
        ner_metrics   = compute_ner_f1(pred_entities, orig_entities)
        results.update(ner_metrics)

        if compute_hallucinations:
            logger.info(f"[{pipeline_label}] Computing Hallucination Rate...")
            halluc_metrics = compute_hallucination_rate(
                generated_summaries=predictions,
                original_texts=self.original_texts,
                ner_fn=self._ner_fn,
            )
            results.update(halluc_metrics)
        else:
            results.update({
                "hallucination_rate":       0.0,
                "total_entities_generated": 0,
                "total_hallucinations":     0,
            })

        timing_metrics = compute_timing_metrics(exec_time_seconds, n)
        results.update(timing_metrics)

        logger.info(
            f"[{pipeline_label}] "
            f"ROUGE-1={results['rouge1']:.4f} | "
            f"ROUGE-L={results['rougeL']:.4f} | "
            f"BERTScore-F1={results['bertscore_f1']:.4f} | "
            f"NER-F1={results['ner_f1']:.4f} | "
            f"Halluc={results['hallucination_rate']:.4f} | "
            f"{results['ms_per_example']:.1f} ms/ex"
        )

        return results

    def compare(self, metrics_a: dict, metrics_b: dict) -> dict:
        # Compares core metric updates and targets between Pipeline A and Pipeline B
        comparison_keys = {
            "rouge1":             "higher_is_better",
            "rouge2":             "higher_is_better",
            "rougeL":             "higher_is_better",
            "bertscore_f1":       "higher_is_better", 
            "ner_f1":             "higher_is_better",
            "hallucination_rate": "lower_is_better",
            "ms_per_example":     "lower_is_better",
        }

        comparison = {}
        for key, direction in comparison_keys.items():
            val_a = metrics_a.get(key, 0.0)
            val_b = metrics_b.get(key, 0.0)
            delta = val_b - val_a

            if direction == "higher_is_better":
                winner = "B" if delta > 0 else ("A" if delta < 0 else "tie")
            else:
                winner = "A" if delta > 0 else ("B" if delta < 0 else "tie")

            comparison[key] = {
                "pipeline_a":      round(val_a, 4),
                "pipeline_b":      round(val_b, 4),
                "delta_B_minus_A": round(delta, 4),
                "winner":          winner,
            }

        return comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    originals = [
        "I have been taking metformin for diabetes for 3 years and recently I started having nausea and stomach pain.",
        "My child has asthma and the doctor prescribed albuterol inhaler but I am worried about side effects.",
    ]
    references = [
        "What are the side effects of metformin for diabetes?",
        "Is albuterol inhaler safe for children with asthma?",
    ]
    predictions_a = [
        "recently started having nausea and stomach pain after taking metformin",
        "doctor prescribed albuterol inhaler worried about side effects",
    ]
    predictions_b = [
        "What are the side effects of metformin for type 2 diabetes?",
        "Are there any side effects of albuterol inhaler for children with asthma?",
    ]

    evaluator = Evaluator(original_texts=originals)

    print("TEST — Evaluator (NER F1 vs original text)")

    metrics_a = evaluator.evaluate(
        predictions=predictions_a,
        references=references,
        pipeline_label="Pipeline_A_Classical",
        exec_time_seconds=0.05,
        compute_hallucinations=False,
    )

    metrics_b = evaluator.evaluate(
        predictions=predictions_b,
        references=references,
        pipeline_label="Pipeline_B_LLM",
        exec_time_seconds=2.3,
        compute_hallucinations=True,
    )

    print("\nComparison Pipeline A vs B:")
    comparison = evaluator.compare(metrics_a, metrics_b)
    for metric, info in comparison.items():
        print(
            f"  {metric:<22} A={info['pipeline_a']:.4f}  B={info['pipeline_b']:.4f}  "
            f"delta={info['delta_B_minus_A']:+.4f}  winner={info['winner']}"
        )