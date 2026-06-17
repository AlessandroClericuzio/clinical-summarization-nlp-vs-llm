"""
evaluation.py — Framework di Valutazione
Progetto NLP: Clinical Summarization — NLP Tradizionale vs LLM

Metriche implementate:
  1. ROUGE-1, ROUGE-2, ROUGE-L     → qualità della sintesi vs Ground Truth
  2. Precision, Recall, F1 (NER)   → conservazione entità cliniche
  3. Hallucination Rate (semantico) → entità generate non ancorate al CHQ
                                      via Cosine Similarity su TF-IDF embeddings
  4. Execution Time                 → ms/esempio e totale

Research Questions coperte:
  RQ1 → ROUGE (qualità semantica Pipeline A vs B)
  RQ2 → F1 NER (accuratezza estrazione entità cliniche)
  RQ3 → Hallucination Rate + Execution Time (trade-off computazionale)
"""

import logging
import time
from typing import Optional

import numpy as np
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────

# Soglia cosine similarity sotto cui un'entità è considerata allucinazione
_HALLUCINATION_THRESHOLD = 0.3

# Tipi ROUGE da calcolare
_ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]


# ─────────────────────────────────────────────
# 1. Metriche ROUGE
# ─────────────────────────────────────────────

def compute_rouge(predictions: list[str],
                  references: list[str]) -> dict:
    """
    Calcola ROUGE-1, ROUGE-2 e ROUGE-L tra predizioni e Ground Truth.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) misura
    l'overlap di n-grammi tra il testo generato e il riferimento:
      - ROUGE-1: overlap unigrammi
      - ROUGE-2: overlap bigrammi
      - ROUGE-L: longest common subsequence

    Args:
        predictions: Lista di sintesi generate dalla pipeline
        references:  Lista di Summary (Ground Truth) dal dataset

    Returns:
        Dizionario con medie aggregate e valori per campione:
          {
            'rouge1': float,  'rouge2': float,  'rougeL': float,
            'rouge1_per_sample': list, 'rouge2_per_sample': list, ...
          }
    """
    scorer = rouge_scorer.RougeScorer(_ROUGE_TYPES, use_stemmer=True)

    per_sample = {rt: [] for rt in _ROUGE_TYPES}

    for pred, ref in zip(predictions, references):
        # Gestione stringhe vuote
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


# ─────────────────────────────────────────────
# 2. Metriche NER — Precision, Recall, F1
# ─────────────────────────────────────────────

def compute_ner_f1(predicted_entities: list[set],
                   reference_entities: list[set]) -> dict:
    """
    Calcola Precision, Recall e F1 per l'estrazione di entità mediche.

    Confronto token-level tra entità estratte dal NER applicato alla
    sintesi generata vs entità estratte dal CHQ originale (reference).

    Logica:
      - TP: entità nella sintesi presenti anche nel CHQ originale
      - FP: entità nella sintesi NON presenti nel CHQ originale
      - FN: entità nel CHQ originale NON presenti nella sintesi

    Args:
        predicted_entities: Lista di set di entità estratte dalla sintesi generata
        reference_entities: Lista di set di entità estratte dal CHQ originale

    Returns:
        Dizionario con precision, recall, f1 aggregati e per campione
    """
    precision_scores = []
    recall_scores    = []
    f1_scores        = []

    for pred_set, ref_set in zip(predicted_entities, reference_entities):
        # Normalizzazione lowercase per confronto case-insensitive
        pred_set = {e.lower().strip() for e in pred_set if e.strip()}
        ref_set  = {e.lower().strip() for e in ref_set  if e.strip()}

        if not pred_set and not ref_set:
            # Entrambi vuoti → accordo perfetto
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            f1_scores.append(1.0)
            continue

        if not pred_set:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
            continue

        if not ref_set:
            # Nessuna entità di riferimento → precision irrilevante, recall = 0
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


# ─────────────────────────────────────────────
# 3. Hallucination Rate (semantico via TF-IDF)
# ─────────────────────────────────────────────

def _semantic_similarity(entity: str, context: str,
                          vectorizer: Optional[TfidfVectorizer] = None) -> float:
    """
    Calcola la Cosine Similarity tra un'entità e il testo di contesto (CHQ).

    Usa TF-IDF per la vettorializzazione.
    Un'entità con similarità < _HALLUCINATION_THRESHOLD viene considerata
    non ancorata al testo originale → allucinazione.

    Args:
        entity:     Testo dell'entità generata dall'LLM
        context:    Testo CHQ originale
        vectorizer: Istanza TfidfVectorizer pre-fittata (opzionale)

    Returns:
        Score di similarità [0.0, 1.0]
    """
    if not entity.strip() or not context.strip():
        return 0.0

    # Corrispondenza esatta come fast path
    if entity.lower().strip() in context.lower():
        return 1.0

    try:
        local_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = local_vec.fit_transform([entity, context])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def compute_hallucination_rate(generated_summaries: list[str],
                                original_texts: list[str],
                                ner_fn) -> dict:
    """
    Calcola il tasso di allucinazione semantica per la Pipeline B (LLM).

    Algoritmo:
      Per ogni sintesi generata:
        1. Estrae le entità mediche dalla sintesi (scispaCy)
        2. Per ogni entità, calcola la Cosine Similarity con il CHQ originale
        3. Entità con sim < threshold → allucinazione
        4. hallucination_rate = allucinazioni / totale_entità_generate

    La scelta semantica (vs match esatto) è più robusta perché cattura
    varianti morfologiche e parafrasie tipiche degli LLM.

    Args:
        generated_summaries: Sintesi generate dalla Pipeline B
        original_texts:      CHQ originali corrispondenti
        ner_fn:              Funzione NER (es. extract_medical_entities da pipeline_classical)

    Returns:
        Dizionario con:
          - 'hallucination_rate': tasso medio [0.0, 1.0]
          - 'total_entities_generated': totale entità generate
          - 'total_hallucinations': totale allucinazioni contate
          - 'hallucination_rate_per_sample': lista per campione
    """
    rates          = []
    total_ents     = 0
    total_halluc   = 0

    for summary, original in zip(generated_summaries, original_texts):
        summary  = summary.strip()  if summary  else ""
        original = original.strip() if original else ""

        if not summary:
            rates.append(0.0)
            continue

        # Estrazione entità dalla sintesi generata
        ner_result = ner_fn(summary)
        entities   = ner_result.get("entity_texts", set())

        if not entities:
            rates.append(0.0)
            continue

        hallucinations = 0
        for entity in entities:
            sim = _semantic_similarity(entity, original)
            if sim < _HALLUCINATION_THRESHOLD:
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


# ─────────────────────────────────────────────
# 4. Metriche Computazionali
# ─────────────────────────────────────────────

def compute_timing_metrics(exec_time_seconds: float,
                            n_samples: int) -> dict:
    """
    Calcola le metriche di efficienza computazionale.

    Args:
        exec_time_seconds: Tempo totale di esecuzione in secondi
        n_samples:         Numero di esempi processati

    Returns:
        Dizionario con tempo totale, medio per esempio e throughput
    """
    ms_per_example = (exec_time_seconds / n_samples * 1000) if n_samples > 0 else 0.0

    return {
        "exec_time_seconds":  round(exec_time_seconds, 3),
        "ms_per_example":     round(ms_per_example, 2),
        "throughput_eps":     round(n_samples / exec_time_seconds, 2) if exec_time_seconds > 0 else 0.0,
    }


# ─────────────────────────────────────────────
# Classe principale: Evaluator
# ─────────────────────────────────────────────

class Evaluator:
    """
    Framework di valutazione unificato per entrambe le pipeline.

    Calcola e aggrega tutte le metriche richieste dal progetto:
      - ROUGE-1, ROUGE-2, ROUGE-L       (RQ1)
      - Precision, Recall, F1 NER       (RQ2)
      - Hallucination Rate semantico     (RQ3 — solo Pipeline B)
      - Execution Time / ms per esempio  (RQ3)

    Esempio d'uso:
        evaluator = Evaluator(original_texts=chq_list)
        metrics = evaluator.evaluate(
            predictions=summaries_a,
            references=ground_truth,
            pipeline_label="Pipeline_A_Classical",
            exec_time_seconds=12.4,
        )
    """

    def __init__(self, original_texts: list[str]):
        """
        Args:
            original_texts: Lista dei CHQ originali.
                            Usati per il calcolo dell'hallucination rate
                            e come reference per il NER.
        """
        self.original_texts = original_texts

        # Import lazy di scispaCy NER per evitare dipendenza circolare
        try:
            from src.NLPPipeline import extract_medical_entities
            self._ner_fn = extract_medical_entities
            logger.info("Evaluator: NER scispaCy disponibile ✓")
        except ImportError:
            self._ner_fn = lambda text: {"entities": [], "entity_texts": set(), "count": 0}
            logger.warning("Evaluator: scispaCy non disponibile, NER disabilitato")

        logger.info(f"Evaluator inizializzato su {len(original_texts)} esempi")

    def _extract_entity_sets(self, texts: list[str]) -> list[set]:
        """
        Estrae i set di entità mediche per una lista di testi.

        Args:
            texts: Lista di testi da analizzare

        Returns:
            Lista di set di testi delle entità
        """
        return [
            self._ner_fn(text).get("entity_texts", set())
            for text in texts
        ]

    def evaluate(self,
                 predictions: list[str],
                 references: list[str],
                 pipeline_label: str,
                 exec_time_seconds: float,
                 compute_hallucinations: bool = True) -> dict:
        """
        Valutazione completa di una pipeline.

        Args:
            predictions:            Lista di sintesi generate dalla pipeline
            references:             Lista di Summary (Ground Truth)
            pipeline_label:         Etichetta identificativa (es. "Pipeline_A_Classical")
            exec_time_seconds:      Tempo totale di esecuzione
            compute_hallucinations: Se True, calcola l'hallucination rate
                                    (rilevante principalmente per Pipeline B)

        Returns:
            Dizionario unificato con tutte le metriche aggregate
        """
        n = len(predictions)
        logger.info(f"[{pipeline_label}] Calcolo metriche su {n} esempi...")

        results = {"pipeline": pipeline_label, "n_samples": n}

        # ── ROUGE ──────────────────────────────────
        logger.info(f"[{pipeline_label}] Calcolo ROUGE...")
        rouge_metrics = compute_rouge(predictions, references)
        results.update(rouge_metrics)

        # ── NER F1 ─────────────────────────────────
        logger.info(f"[{pipeline_label}] Calcolo NER F1...")
        pred_entities = self._extract_entity_sets(predictions)
        ref_entities  = self._extract_entity_sets(self.original_texts)
        ner_metrics   = compute_ner_f1(pred_entities, ref_entities)
        results.update(ner_metrics)

        # ── Hallucination Rate ──────────────────────
        if compute_hallucinations:
            logger.info(f"[{pipeline_label}] Calcolo Hallucination Rate (semantico)...")
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

        # ── Timing ─────────────────────────────────
        timing_metrics = compute_timing_metrics(exec_time_seconds, n)
        results.update(timing_metrics)

        # ── Log riepilogo ───────────────────────────
        logger.info(
            f"[{pipeline_label}] "
            f"ROUGE-1={results['rouge1']:.4f} | "
            f"ROUGE-L={results['rougeL']:.4f} | "
            f"NER-F1={results['ner_f1']:.4f} | "
            f"Halluc={results['hallucination_rate']:.4f} | "
            f"{results['ms_per_example']:.1f} ms/ex"
        )

        return results

    def compare(self,
                metrics_a: dict,
                metrics_b: dict) -> dict:
        """
        Confronta le metriche di Pipeline A e Pipeline B.
        Calcola delta e indica quale pipeline è migliore per ogni metrica.

        Args:
            metrics_a: Risultati Pipeline A
            metrics_b: Risultati Pipeline B

        Returns:
            Dizionario con delta e winner per ogni metrica chiave
        """
        comparison_keys = {
            "rouge1":             "higher_is_better",
            "rouge2":             "higher_is_better",
            "rougeL":             "higher_is_better",
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
                "pipeline_a": round(val_a, 4),
                "pipeline_b": round(val_b, 4),
                "delta_B_minus_A": round(delta, 4),
                "winner": winner,
            }

        return comparison


# ─────────────────────────────────────────────
# Test rapido (eseguibile direttamente)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Dati di esempio per test
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

    print("\n" + "="*60)
    print("TEST — Evaluator")
    print("="*60)

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

    print("\n📊 Confronto Pipeline A vs B:")
    comparison = evaluator.compare(metrics_a, metrics_b)
    for metric, info in comparison.items():
        print(f"  {metric:<22} A={info['pipeline_a']:.4f}  B={info['pipeline_b']:.4f}  "
              f"Δ={info['delta_B_minus_A']:+.4f}  winner={info['winner']}")
