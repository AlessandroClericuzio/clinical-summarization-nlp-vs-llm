# Clinical Summarization - PubMed Dataset

Confronto tra due pipeline di summarization applicate ad articoli scientifici biomedici (dataset PubMed): una pipeline classica basata su TextRank e una pipeline basata su LLM (via Ollama).

## Struttura del progetto

- `main.py` - script principale: carica il dataset, esegue entrambe le pipeline, calcola le metriche e salva i risultati.
- `src/NLPPipeline.py` - Pipeline A (classica/estrattiva): TextRank + bias di posizione, tramite spaCy/scispaCy.
- `src/LLMPipeline.py` - Pipeline B (LLM/astrattiva): genera riassunti tramite modello LLM servito da Ollama, con diverse strategie di prompting.
- `src/evaluation.py` - modulo di valutazione: ROUGE, BERTScore, NER F1, hallucination rate e metriche di tempo.

## Requisiti

- Python 3.10+
- Un server Ollama attivo e raggiungibile (di default su `http://localhost:11434`) con il modello LLM scaricato
- Modelli spaCy/scispaCy installati:
  - `en_core_web_sm`
  - `en_ner_bc5cdr_md` (scispaCy, per il riconoscimento di entità biomediche)
- Dataset preprocessato disponibile in `data/pubmed_cleaned.csv` (generato tramite un notebook di esplorazione, non incluso in questo repo)

Librerie Python principali: `pandas`, `numpy`, `spacy`, `scikit-learn`, `rouge-score`, `bert-score`, `requests`.

## Cosa scaricare prima di lanciare il progetto

Prima di eseguire `main.py` è necessario scaricare/installare:

### 1. Dipendenze Python

```bash
pip install pandas numpy spacy scikit-learn rouge-score bert-score requests
```

### 2. Modelli spaCy / scispaCy

```bash
python -m spacy download en_core_web_sm
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```

### 3. Ollama e il modello LLM

- Installare Ollama seguendo le istruzioni ufficiali: https://ollama.com
- Avviare il servizio Ollama (di default esposto su `http://localhost:11434`)
- Scaricare il modello LLM desiderato, ad esempio:

```bash
ollama pull llama3.1:8b
```

### 4. Dataset

- Il dataset preprocessato deve essere disponibile in `data/pubmed_cleaned.csv`.
- In alternativa, è possibile specificare direttamente un dataset Hugging Face tramite il parametro `--dataset_name` (es. `ccdv/pubmed-summarization`), che verrà scaricato automaticamente al primo utilizzo.

## Come si esegue

Esempio base:

```bash
python main.py --llm_model qwen3.5:27b --prompting few-shot --sample_size 500
```

Per lanciarlo con il dataset Hugging Face `ccdv/pubmed-summarization` e il modello `llama3.1:8b`, eseguire le pipeline con le diverse strategie di prompting:

```bash
python main.py --dataset_name ccdv/pubmed-summarization --sample_size 500 --llm_model llama3.1:8b --prompting zero-shot
python main.py --dataset_name ccdv/pubmed-summarization --sample_size 500 --llm_model llama3.1:8b --prompting one-shot
python main.py --dataset_name ccdv/pubmed-summarization --sample_size 500 --llm_model llama3.1:8b --prompting few-shot
```

### Parametri principali

| Parametro | Default | Descrizione |
|---|---|---|
| `--sample_size` | 500 | numero di campioni da usare (omettere per usarli tutti) |
| `--output_dir` | results/ | cartella di output |
| `--llm_model` | qwen3.5:27b | nome del modello Ollama da usare |
| `--prompting` | few-shot | strategia di prompting: zero-shot, one-shot, few-shot, cot |
| `--temperature` | 0.1 | temperatura di generazione dell'LLM |
| `--max_new_tokens` | 2048 | numero massimo di token generati |
| `--position_bias_weight` | 0.25 | peso del bias di posizione nella pipeline classica (0.0-1.0) |
| `--dataset_name` | - | nome del dataset Hugging Face da usare (es. `ccdv/pubmed-summarization`), in alternativa al CSV locale |

## Cosa produce

Nella cartella di output (`results/` di default):

- `predictions_<prompting>.csv` - dataset originale con le colonne `summary_classical` e `summary_llm` aggiunte
- `metrics_summary_<prompting>.csv` - tabella riassuntiva delle metriche per entrambe le pipeline

## Metriche calcolate

- **ROUGE-1 / ROUGE-2 / ROUGE-L** - sovrapposizione lessicale con il riassunto di riferimento
- **BERTScore** (precision, recall, F1) - similarità semantica basata su embedding contestuali
- **NER F1** - confronto tra le entità mediche estratte dal riassunto e quelle del testo originale
- **Hallucination rate** - percentuale di entità nel riassunto non ancorate al testo sorgente (calcolata solo per la pipeline LLM)
- **Metriche di tempo** - tempo di esecuzione totale, ms per esempio, throughput

## Note

- La pipeline classica (Pipeline A) è deterministica e non richiede servizi esterni.
- La pipeline LLM (Pipeline B) richiede che Ollama sia attivo e raggiungibile; in caso contrario lo script termina con un errore esplicito.
- Se scispaCy non è installato, il calcolo delle metriche NER e hallucination viene disabilitato automaticamente (fallback silenzioso, con warning nei log).