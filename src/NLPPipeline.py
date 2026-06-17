"""
pipeline_classical.py — Pipeline A: NLP Classica (Estrattiva)
Progetto NLP: Clinical Summarization — NLP Tradizionale vs LLM

Approccio:
  1. Preprocessing     → pulizia, tokenizzazione, lemmatizzazione (spaCy)
  2. POS Tagging       → identificazione token rilevanti
  3. Sintesi Estrattiva → TF-IDF + Cosine Similarity + TextRank
  4. NER               → estrazione entità mediche con scispaCy

Tecniche del corso utilizzate:
  - Tokenizzazione, stop-words, lemmatizzazione  (Text Normalization)
  - POS Tagging, Dependency Parsing              (Corpus Preprocessing)
  - TF-IDF, Cosine Similarity                   (Vector Semantics)
  - TextRank (graph-based)                       (Text Summarization — Metodi Estrattivi)
  - NER                                          (Information Extraction)
"""

import re
import logging
import warnings
from typing import Optional

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Caricamento modelli spaCy / scispaCy
# ─────────────────────────────────────────────

def _load_spacy_model(model_name: str) -> Optional[spacy.Language]:
    """Carica un modello spaCy con fallback graceful."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning(
            f"Modello spaCy '{model_name}' non trovato. "
            f"Installalo con: python -m spacy download {model_name}"
        )
        return None


# Modello general-purpose per preprocessing e POS tagging
_NLP_GENERAL = _load_spacy_model("en_core_web_sm") 

# Modello biomedico per NER clinica
_NLP_SCI = _load_spacy_model("en_ner_bc5cdr_md")
if _NLP_SCI is None:
    logger.warning(
        "scispaCy 'en_core_sci_sm' non disponibile. "
        "Installa con: pip install scispacy && "
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz"
    )


# ─────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────


# POS tag rilevanti per filtrare token significativi
_RELEVANT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}

# Soglia minima di similarità per aggiungere un arco nel grafo TextRank
_SIMILARITY_THRESHOLD = 0.0

# Numero di iterazioni PageRank
_PAGERANK_ITERATIONS = 100
_PAGERANK_DAMPING = 0.85

# Numero minimo di frasi per applicare TextRank
_MIN_SENTENCES = 2


# ─────────────────────────────────────────────
# Step 1 — Preprocessing
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Pulisce il testo grezzo del CHQ.
    Normalizza spazi e caratteri speciali.

    Args:
        text: Testo grezzo CHQ

    Returns:
        Testo pulito
    """

    # Rimozione URL
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Normalizzazione spazi multipli e newline
    text = re.sub(r"[ \t]+", " ", text)          # normalizza spazi orizzontali
    text = re.sub(r"\s*\n\s*", ". ", text)       # converte newline in punto (aiuta spaCy)
    text = re.sub(r"\.\.+", ".", text)           # rimuove punti doppi generati
    text = text.strip()

    return text


def lemmatize_sentence(sentence: str, nlp: spacy.Language) -> str:
    """
    Tokenizza, rimuove stop-words e lemmatizza una frase.
    Mantiene solo token con POS tag rilevanti (NOUN, PROPN, VERB, ADJ).

    Args:
        sentence: Frase originale
        nlp: Modello spaCy caricato

    Returns:
        Stringa di lemmi filtrati
    """
    doc = nlp(sentence.lower())
    lemmas = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and token.pos_ in _RELEVANT_POS
        and len(token.lemma_) > 1
    ]
    return " ".join(lemmas)


def split_into_sentences(text: str, nlp: spacy.Language) -> list[str]:
    """
    Segmenta il testo in frasi usando il sentence segmenter di spaCy.
    Filtra frasi troppo corte (< 3 token) che non portano informazione utile.

    Args:
        text: Testo pulito
        nlp: Modello spaCy

    Returns:
        Lista di frasi originali (non lemmatizzate)
    """
    doc = nlp(text)
    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.strip().split()) >= 2
    ]
    return sentences


# ─────────────────────────────────────────────
# Step 2 — POS Tagging e analisi struttura
# ─────────────────────────────────────────────

def extract_pos_features(text: str, nlp: spacy.Language) -> dict:
    """
    Estrae feature grammaticali dal testo tramite POS Tagging
    e Dependency Parsing (spaCy).

    Identifica il nucleo informativo della frase:
    soggetti, radici verbali e oggetti diretti.

    Args:
        text: Testo originale
        nlp: Modello spaCy

    Returns:
        Dizionario con liste di token per categoria grammaticale
    """
    doc = nlp(text)

    features = {
        "nouns":     [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")],
        "verbs":     [t.text for t in doc if t.pos_ == "VERB"],
        "adjectives":[t.text for t in doc if t.pos_ == "ADJ"],
        "subjects":  [t.text for t in doc if t.dep_ in ("nsubj", "nsubjpass")],
        "roots":     [t.text for t in doc if t.dep_ == "ROOT"],
        "objects":   [t.text for t in doc if t.dep_ in ("dobj", "pobj", "attr")],
    }

    return features


# ─────────────────────────────────────────────
# Step 3 — Sintesi Estrattiva: TextRank
# ─────────────────────────────────────────────

def build_tfidf_matrix(sentences: list[str], lemmatized: list[str]) -> np.ndarray:
    """
    Vettorializza le frasi con TF-IDF.
    Usa le versioni lemmatizzate per il calcolo ma mantiene
    le frasi originali come output finale.

    Args:
        sentences:   Liste di frasi originali (usate come fallback)
        lemmatized:  Liste di frasi lemmatizzate (usate per TF-IDF)

    Returns:
        Matrice TF-IDF (n_sentences × n_features)
    """
    # Filtra frasi lemmatizzate vuote: usa la frase originale come fallback
    corpus = [
        lem if lem.strip() else orig
        for lem, orig in zip(lemmatized, sentences)
    ]

    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),   # unigrammi + bigrammi
        sublinear_tf=True,    # log(tf) per appiattire le frequenze molto alte
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
        return matrix.toarray()
    except ValueError:
        # Corpus troppo piccolo o vuoto
        return np.zeros((len(sentences), 1))


def build_similarity_graph(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Costruisce la matrice di similarità tra frasi tramite Cosine Similarity.
    Ogni cella (i,j) rappresenta quanto le frasi i e j siano simili.

    Args:
        tfidf_matrix: Matrice TF-IDF (n_sentences × n_features)

    Returns:
        Matrice di similarità (n_sentences × n_sentences)
    """
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Azzera la diagonale (similarità di una frase con se stessa)
    np.fill_diagonal(similarity_matrix, 0.0)

    # Azzera valori sotto soglia (rimuove archi deboli dal grafo)
    similarity_matrix[similarity_matrix < _SIMILARITY_THRESHOLD] = 0.0

    return similarity_matrix


def pagerank(similarity_matrix: np.ndarray,
             damping: float = _PAGERANK_DAMPING,
             iterations: int = _PAGERANK_ITERATIONS) -> np.ndarray:
    """
    Implementazione di PageRank per il ranking delle frasi (TextRank).

    Algoritmo:
        scores_new[i] = (1 - d) + d * Σ_j (sim[j,i] / Σ_k sim[j,k]) * scores[j]

    dove d è il damping factor (tipicamente 0.85).

    Args:
        similarity_matrix: Matrice di similarità tra frasi
        damping:           Damping factor (probabilità di seguire un arco)
        iterations:        Numero di iterazioni

    Returns:
        Vettore di scores per ogni frase
    """
    n = similarity_matrix.shape[0]

    # Normalizzazione per riga: trasforma in matrice di transizione
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    # Crea la matrice di transizione
    # Se row_sum > 0 dividi, altrimenti distribuisci uniformemente (1/n)
    with np.errstate(divide='ignore', invalid='ignore'):
        transition_matrix = np.where(row_sums > 0, similarity_matrix / row_sums, 1.0 / n)
    # Inizializzazione uniforme degli scores
    scores = np.ones(n) / n

    # Iterazione PageRank
    for _ in range(iterations):
        new_scores = (1 - damping) / n + damping * transition_matrix.T @ scores
        # Verifica convergenza
        if np.linalg.norm(new_scores - scores) < 1e-6:
            break
        scores = new_scores

    return scores


def textrank_summarize(text: str, nlp: spacy.Language, n_sentences: int = 1) -> str:
    """
    Pipeline completa di sintesi estrattiva con TextRank.

    Flusso:
        testo pulito → segmentazione → lemmatizzazione →
        TF-IDF → Cosine Similarity → grafo → PageRank → frase top

    Args:
        text:        Testo CHQ preprocessato
        nlp:         Modello spaCy
        n_sentences: Numero di frasi da estrarre (default=1, derivato da compression ratio 6.6×)

    Returns:
        Stringa con le frasi più rilevanti estratte
    """
    # Segmentazione in frasi
    sentences = split_into_sentences(text, nlp)

    # Caso degenere: testo con meno frasi del minimo richiesto
    if len(sentences) == 0:
        return text.strip()
    if len(sentences) < _MIN_SENTENCES:
        return sentences[0].strip()  # restituisce la prima frase se è l'unica disponibile

    # Lemmatizzazione per la rappresentazione vettoriale
    lemmatized = [lemmatize_sentence(sent, nlp) for sent in sentences]

    # Vettorializzazione TF-IDF
    tfidf_matrix = build_tfidf_matrix(sentences, lemmatized)

    # Grafo di similarità (Cosine Similarity)
    similarity_matrix = build_similarity_graph(tfidf_matrix)

    # Ranking frasi con PageRank (TextRank)
    scores = pagerank(similarity_matrix)

    # Selezione delle top-n frasi con score massimo
    seen_texts = set()
    top_indices = []
    for idx in np.argsort(scores)[::-1]:
        sentence = sentences[idx].strip()
        if sentence not in seen_texts:
            seen_texts.add(sentence)
            top_indices.append(idx)
        if len(top_indices) == n_sentences:
            break

    top_indices = sorted(top_indices)
    summary = " ".join(sentences[i] for i in top_indices)
    return summary.strip()


# ─────────────────────────────────────────────
# Step 4 — NER con scispaCy
# ─────────────────────────────────────────────

def extract_medical_entities(text: str) -> dict:
    """
    Estrae entità mediche dal testo usando scispaCy (en_ner_bc5cdr_m).

    Le entità estratte includono: farmaci, patologie, procedure,
    sostanze chimiche, anatomia (dipende dal modello scispaCy caricato).

    Args:
        text: Testo da analizzare (tipicamente il CHQ originale)

    Returns:
        Dizionario con:
          - 'entities': lista di (testo_entità, label)
          - 'entity_texts': set di testi delle entità (per confronto)
          - 'count': numero totale di entità trovate
    """
    if _NLP_SCI is None:
        # Fallback: nessuna entità se scispaCy non è disponibile
        return {"entities": [], "entity_texts": set(), "count": 0}

    doc = _NLP_SCI(text)

    entities = [
        (ent.text.lower().strip(), ent.label_)
        for ent in doc.ents
        if len(ent.text.strip()) > 1
    ]

    entity_texts = {e[0] for e in entities}

    return {
        "entities":     entities,
        "entity_texts": entity_texts,
        "count":        len(entities),
    }


# ─────────────────────────────────────────────
# Classe principale: ClassicalPipeline
# ─────────────────────────────────────────────

class ClassicalPipeline:
    """
    Pipeline A: NLP Classica Estrattiva.

    Combina preprocessing testuale, POS Tagging, TextRank
    per la sintesi estrattiva e scispaCy per il NER clinico.

    Esempio d'uso:
        pipeline = ClassicalPipeline()
        summary = pipeline.run("I have been suffering from...")
        results = pipeline.run_batch(list_of_texts)
    """

    def __init__(self, n_summary_sentences: int = 3):
        """
        Args:
          # n_summary_sentences=3: motivato dalla lunghezza media degli abstract PubMed (~200 parole)
        """
        self.n_summary_sentences = n_summary_sentences

        if _NLP_GENERAL is None:
            raise RuntimeError(
                "Modello spaCy 'en_core_web_sm' non trovato. "
                "Installalo con: python -m spacy download en_ner_bc5cdr_md"
            )

        self.nlp = _NLP_GENERAL
        logger.info(
            f"ClassicalPipeline inizializzata | "
            f"modello: en_core_web_sm | "
            f"n_summary_sentences: {self.n_summary_sentences} | "
            f"scispaCy NER: {'✓' if _NLP_SCI else '✗ (non disponibile)'}"
        )

    def run(self, text: str) -> str:
        """
        Esegue la pipeline completa su un singolo testo CHQ.

        Step:
            1. Preprocessing (pulizia, normalizzazione)
            2. POS Tagging (analisi struttura grammaticale)
            3. TextRank (sintesi estrattiva)
            4. NER (estrazione entità mediche)

        Args:
            text: Testo grezzo CHQ

        Returns:
            Stringa con la sintesi estratta
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # Step 1 — Preprocessing
        clean_text = preprocess_text(text)

        if not clean_text.strip():
            return ""

        # Step 2 — POS Tagging (analisi, non modifica il testo)
        # I risultati sono disponibili ma non bloccano la pipeline
        _ = extract_pos_features(clean_text, self.nlp)

        # Step 3 — Sintesi Estrattiva (TextRank)
        summary = textrank_summarize(
            clean_text,
            self.nlp,
            n_sentences=self.n_summary_sentences,
        )

        return summary

    def run_with_ner(self, text: str) -> dict:
        """
        Versione estesa di run() che include anche il NER.

        Args:
            text: Testo grezzo CHQ

        Returns:
            Dizionario con:
              - 'summary': sintesi estrattiva
              - 'ner_original': entità estratte dal CHQ originale
              - 'ner_summary': entità estratte dalla sintesi generata
              - 'pos_features': feature grammaticali POS
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "summary":      "",
                "ner_original": {"entities": [], "entity_texts": set(), "count": 0},
                "ner_summary":  {"entities": [], "entity_texts": set(), "count": 0},
                "pos_features": {},
            }

        # Step 1 — Preprocessing
        clean_text = preprocess_text(text)

        # Step 2 — POS Tagging
        pos_features = extract_pos_features(clean_text, self.nlp)

        # Step 3 — Sintesi Estrattiva
        summary = textrank_summarize(
            clean_text,
            self.nlp,
            n_sentences=self.n_summary_sentences,
        )

        # Step 4 — NER sul testo originale e sulla sintesi
        ner_original = extract_medical_entities(clean_text)
        ner_summary  = extract_medical_entities(summary)

        return {
            "summary":      summary,
            "ner_original": ner_original,
            "ner_summary":  ner_summary,
            "pos_features": pos_features,
        }

    def run_batch(self, texts: list[str]) -> list[str]:
        """
        Esegue la pipeline su una lista di testi CHQ.
        Usato dall'orchestratore main.py per il confronto con Pipeline B.

        Args:
            texts: Lista di testi CHQ grezzi

        Returns:
            Lista di sintesi estrattive (stessa lunghezza dell'input)
        """
        summaries = []
        for i, text in enumerate(texts):
            summary = self.run(text)
            summaries.append(summary)

            if (i + 1) % 50 == 0:
                logger.info(f"  Pipeline A: {i + 1}/{len(texts)} esempi processati")

        return summaries

    def run_batch_with_ner(self, texts: list[str]) -> list[dict]:
        """
        Versione estesa di run_batch() con NER incluso.
        Usata da evaluation.py per le metriche di Information Extraction.

        Args:
            texts: Lista di testi CHQ grezzi

        Returns:
            Lista di dizionari con summary + NER
        """
        results = []
        for i, text in enumerate(texts):
            result = self.run_with_ner(text)
            results.append(result)

            if (i + 1) % 50 == 0:
                logger.info(f"  Pipeline A (NER): {i + 1}/{len(texts)} esempi processati")

        return results


# ─────────────────────────────────────────────
# Test rapido (eseguibile direttamente)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    sample_chq = """
    SUBJECT: Questions about my medication
    MESSAGE: Hello, I am a 45 year old woman and I have been diagnosed with type 2 diabetes
    about 3 years ago. My doctor has prescribed me metformin 500mg twice a day. Recently
    I started experiencing some side effects like nausea and stomach pain. I also have
    high blood pressure and I take lisinopril for that. My question is whether it is safe
    to continue taking metformin with these side effects, or if there is an alternative
    medication I could ask my doctor about. I am also wondering if my blood pressure
    medication could be interacting with the metformin. Thank you for your help.
    """

    pipeline = ClassicalPipeline(n_summary_sentences=1)

    print("\n" + "="*60)
    print("TEST — Pipeline A (NLP Classica)")
    print("="*60)
    print(f"\nCHQ originale ({len(sample_chq.split())} parole):")
    print(sample_chq.strip())

    result = pipeline.run_with_ner(sample_chq)

    print(f"\n✅ Sintesi estrattiva ({len(result['summary'].split())} parole):")
    print(result["summary"])

    print(f"\n🔬 Entità mediche nel CHQ originale ({result['ner_original']['count']}):")
    for ent_text, ent_label in result["ner_original"]["entities"]:
        print(f"   [{ent_label}] {ent_text}")

    print(f"\n🔬 Entità mediche nella sintesi ({result['ner_summary']['count']}):")
    for ent_text, ent_label in result["ner_summary"]["entities"]:
        print(f"   [{ent_label}] {ent_text}")

    print(f"\n📊 POS features:")
    for tag, tokens in result["pos_features"].items():
        if tokens:
            print(f"   {tag}: {tokens[:5]}")