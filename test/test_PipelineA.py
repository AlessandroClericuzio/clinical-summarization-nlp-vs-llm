"""
test_pipeline_a.py — Test rapido per Pipeline A (Estrattiva)
Carica un piccolo campione da PubMed e verifica il funzionamento di TextRank + NER.
"""

"""
test_pipeline_a.py — Test rapido per Pipeline A (Estrattiva)
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import logging
import time
from datasets import load_dataset
from src.NLPPipeline import ClassicalPipeline
from src.evaluation import compute_rouge

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # 1. Carica un piccolo campione dal dataset
    logger.info("Caricamento dataset PubMed (solo 10 esempi per test)...")
    ds = load_dataset("ccdv/pubmed-summarization", split="train")
    
    # Prendi i primi 10 esempi (o campiona casualmente)
    sample_size = 10
    articles = ds["article"][:sample_size]
    abstracts = ds["abstract"][:sample_size]
    logger.info(f"Caricati {len(articles)} articoli.")

    # 2. Inizializza Pipeline A
    # Usiamo 3 frasi per ottenere un riassunto di lunghezza paragonabile all'abstract (~60 parole)
    logger.info("Inizializzazione ClassicalPipeline (n_summary_sentences=3)...")
    pipeline_a = ClassicalPipeline(n_summary_sentences=3)

    # 3. Esegui la pipeline e misura il tempo
    logger.info("Esecuzione TextRank su tutti gli articoli...")
    start_time = time.perf_counter()
    summaries = pipeline_a.run_batch(articles)
    elapsed = time.perf_counter() - start_time

    logger.info(f"Completato in {elapsed:.2f}s ({elapsed/sample_size*1000:.1f} ms/esempio)")

    # 4. Stampa risultati per ogni esempio
    print("\n" + "="*80)
    print(f"🔬 TEST PIPELINE A — {sample_size} esempi")
    print("="*80)

    for i, (article, summary, ref) in enumerate(zip(articles, summaries, abstracts)):
        print(f"\n{'─'*80}")
        print(f"📄 Esempio {i+1}")
        print(f"{'─'*80}")
        print(f"📝 ARTICOLO (prime 400 caratteri):\n{article[:400]}...\n")
        print(f"📌 RIASSUNTO ESTRATTO (TextRank):\n{summary}\n")
        print(f"✅ GROUND TRUTH (Abstract):\n{ref}\n")

    # 5. (Opzionale) Calcola ROUGE per avere un'anteprima delle metriche
    logger.info("Calcolo ROUGE (anteprima metriche)...")
    rouge_scores = compute_rouge(summaries, abstracts)
    
    print(f"\n{'─'*80}")
    print("📊 METRICHE ROUGE (anteprima su 10 esempi):")
    print(f"{'─'*80}")
    print(f"  ROUGE-1  : {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2  : {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L  : {rouge_scores['rougeL']:.4f}")
    print(f"  Tempo medio: {elapsed/sample_size*1000:.1f} ms/esempio")
    print("="*80)

if __name__ == "__main__":
    main()