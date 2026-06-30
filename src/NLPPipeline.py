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


def _load_spacy_model(model_name: str) -> Optional[spacy.Language]:
    try:
        return spacy.load(model_name)
    except OSError:
        logger.warning(
            f"spaCy model '{model_name}' not found. "
            f"Install it using: python -m spacy download {model_name}"
        )
        return None


_NLP_GENERAL = _load_spacy_model("en_core_web_sm")
_NLP_SCI = _load_spacy_model("en_ner_bc5cdr_md")

if _NLP_SCI is None:
    logger.warning(
        "scispaCy model 'en_ner_bc5cdr_md' is not available. "
        "Please install scispacy and download the target model."
    )


_RELEVANT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}
_SIMILARITY_THRESHOLD = 0.0
_PAGERANK_ITERATIONS = 100
_PAGERANK_DAMPING = 0.85
_MIN_SENTENCES = 2
_DEFAULT_N_SENTENCES = 5
_DEFAULT_POSITION_BIAS = 0.25


def preprocess_text(text: str) -> str:
    # Clean raw text by removing URLs and normalizing layout/spacing
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", ". ", text)
    text = re.sub(r"\.\.+", ".", text)
    return text.strip()


def lemmatize_sentence(sentence: str, nlp: spacy.Language) -> str:
    # Tokenize, filter significant POS tokens, and extract alphanumeric lemmas
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
    # Segment document into clean sentences while filtering noise tokens
    doc = nlp(text)
    seen = set()
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        if len(s.split()) >= 5 and s not in seen:
            seen.add(s)
            sentences.append(s)
    return sentences


def extract_pos_features(text: str, nlp: spacy.Language) -> dict:
    # Identify key syntactic features and structural grammatical entities
    doc = nlp(text)
    return {
        "nouns":      [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")],
        "verbs":      [t.text for t in doc if t.pos_ == "VERB"],
        "adjectives": [t.text for t in doc if t.pos_ == "ADJ"],
        "subjects":   [t.text for t in doc if t.dep_ in ("nsubj", "nsubjpass")],
        "roots":      [t.text for t in doc if t.dep_ == "ROOT"],
        "objects":    [t.text for t in doc if t.dep_ in ("dobj", "pobj", "attr")],
    }


def build_tfidf_matrix(sentences: list[str], lemmatized: list[str]) -> np.ndarray:
    # Vectorize document corpus using word unigrams and bigram combinations
    corpus = [
        lem if lem.strip() else orig
        for lem, orig in zip(lemmatized, sentences)
    ]
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
        return matrix.toarray()
    except ValueError:
        return np.zeros((len(sentences), 1))


def build_similarity_graph(tfidf_matrix: np.ndarray) -> np.ndarray:
    # Construct complete graph adjacency matrix using calculated cosine metrics
    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0.0)
    similarity_matrix[similarity_matrix < _SIMILARITY_THRESHOLD] = 0.0
    return similarity_matrix


def pagerank(similarity_matrix: np.ndarray,
             damping: float = _PAGERANK_DAMPING,
             iterations: int = _PAGERANK_ITERATIONS) -> np.ndarray:
    # Compute PageRank importance distribution over the sentence graph
    n = similarity_matrix.shape[0]
    row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        transition_matrix = np.where(row_sums > 0, similarity_matrix / row_sums, 1.0 / n)

    scores = np.ones(n) / n
    for _ in range(iterations):
        new_scores = (1 - damping) / n + damping * transition_matrix.T @ scores
        if np.linalg.norm(new_scores - scores) < 1e-6:
            break
        scores = new_scores
    return scores


def _position_bias_vector(n: int, bias_weight: float) -> np.ndarray:
    # Apply a linear scoring offset favoring information placed later in documents
    if n == 0 or bias_weight == 0.0:
        return np.zeros(n)
    return np.linspace(0.0, bias_weight, n)


def textrank_summarize(
    text: str,
    nlp: spacy.Language,
    n_sentences: int = _DEFAULT_N_SENTENCES,
    position_bias_weight: float = _DEFAULT_POSITION_BIAS,
) -> str:
    # Select best distinct summary candidates matching target score intersections
    sentences = split_into_sentences(text, nlp)

    if len(sentences) == 0:
        return text.strip()
    if len(sentences) < _MIN_SENTENCES:
        return sentences[0].strip()

    lemmatized = [lemmatize_sentence(sent, nlp) for sent in sentences]
    tfidf_matrix = build_tfidf_matrix(sentences, lemmatized)
    similarity_matrix = build_similarity_graph(tfidf_matrix)
    scores = pagerank(similarity_matrix)

    score_range = scores.max() - scores.min()
    scores_norm = (scores - scores.min()) / score_range if score_range > 0 else scores.copy()

    bias = _position_bias_vector(len(sentences), position_bias_weight)
    combined = scores_norm * (1.0 - position_bias_weight) + bias

    seen_texts = set()
    top_indices = []
    for idx in np.argsort(combined)[::-1]:
        sentence = sentences[idx].strip()
        if sentence not in seen_texts:
            seen_texts.add(sentence)
            top_indices.append(idx)
        if len(top_indices) == n_sentences:
            break

    top_indices = sorted(top_indices)
    return " ".join(sentences[i] for i in top_indices).strip()


def extract_medical_entities(text: str) -> dict:
    # Extract entities and medical target labels using customized scispaCy profiles
    if _NLP_SCI is None:
        return {"entities": [], "entity_texts": set(), "count": 0}

    doc = _NLP_SCI(text)
    entities = [
        (ent.text.lower().strip(), ent.label_)
        for ent in doc.ents
        if len(ent.text.strip()) > 1
    ]
    return {
        "entities":     entities,
        "entity_texts": {e[0] for e in entities},
        "count":        len(entities),
    }


class ClassicalPipeline:

    def __init__(
        self,
        n_summary_sentences: int = _DEFAULT_N_SENTENCES,
        position_bias_weight: float = _DEFAULT_POSITION_BIAS,
    ):
        self.n_summary_sentences = n_summary_sentences
        self.position_bias_weight = position_bias_weight

        if _NLP_GENERAL is None:
            raise RuntimeError("spaCy model 'en_core_web_sm' not found. Please install it.")

        self.nlp = _NLP_GENERAL
        ner_status = "available" if _NLP_SCI else "not available"
        logger.info(
            f"ClassicalPipeline initialized - model: en_core_web_sm - "
            f"n_summary_sentences: {self.n_summary_sentences} - "
            f"position_bias_weight: {self.position_bias_weight} - "
            f"scispaCy NER: {ner_status}"
        )

    def run(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        clean_text = preprocess_text(text)
        if not clean_text.strip():
            return ""

        return textrank_summarize(
            clean_text,
            self.nlp,
            n_sentences=self.n_summary_sentences,
            position_bias_weight=self.position_bias_weight,
        )

    def run_with_ner(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {
                "summary":      "",
                "ner_original": {"entities": [], "entity_texts": set(), "count": 0},
                "ner_summary":  {"entities": [], "entity_texts": set(), "count": 0},
                "pos_features": {},
            }

        clean_text = preprocess_text(text)
        pos_features = extract_pos_features(clean_text, self.nlp)

        summary = textrank_summarize(
            clean_text,
            self.nlp,
            n_sentences=self.n_summary_sentences,
            position_bias_weight=self.position_bias_weight,
        )

        return {
            "summary":      summary,
            "ner_original": extract_medical_entities(clean_text),
            "ner_summary":  extract_medical_entities(summary),
            "pos_features": pos_features,
        }

    def run_batch(self, texts: list[str]) -> list[str]:
        summaries = []
        for i, text in enumerate(texts):
            summaries.append(self.run(text))
            if (i + 1) % 50 == 0:
                logger.info(f"Pipeline A: {i + 1}/{len(texts)} samples processed")
        return summaries

    def run_batch_with_ner(self, texts: list[str]) -> list[dict]:
        results = []
        for i, text in enumerate(texts):
            results.append(self.run_with_ner(text))
            if (i + 1) % 50 == 0:
                logger.info(f"Pipeline A (NER): {i + 1}/{len(texts)} samples processed")
        return results