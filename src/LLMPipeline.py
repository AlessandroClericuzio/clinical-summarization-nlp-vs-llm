"""
LLMPipeline.py — Pipeline B: Generative LLM (Abstractive)
Progetto NLP: Clinical Summarization — NLP Tradizionale vs LLM

Supporta:
  - Qualsiasi modello disponibile su Ollama (es. mixtral, mistral, llama3, ecc.)
  - Strategie di prompting: zero-shot, few-shot, chain-of-thought (CoT)
  - Backend: Ollama (HTTP API locale)
"""

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Esempi fissi per few-shot (presi da PubMed per essere in dominio)
FEW_SHOT_EXAMPLES = [
    {
        "article": (
            "BACKGROUND: The efficacy of acupuncture for chronic low back pain "
            "has been controversial. OBJECTIVE: To evaluate the effectiveness of "
            "acupuncture vs sham acupuncture and no acupuncture for chronic low "
            "back pain. METHODS: A randomized controlled trial with 638 patients "
            "was conducted. RESULTS: Acupuncture was significantly more effective "
            "than no acupuncture, but not more than sham acupuncture. "
            "CONCLUSIONS: Acupuncture is effective for chronic low back pain, "
            "but the effect size is small and may be due to placebo."
        ),
        "summary": (
            "Acupuncture shows small but significant benefit over no treatment "
            "for chronic low back pain, but not over sham acupuncture, suggesting "
            "a possible placebo effect."
        )
    },
    {
        "article": (
            "BACKGROUND: The role of vitamin D supplementation in preventing "
            "fractures is unclear. OBJECTIVE: To determine whether vitamin D "
            "supplementation reduces the risk of fractures in older adults. "
            "METHODS: We conducted a meta-analysis of 11 randomized controlled "
            "trials involving 31,022 participants. RESULTS: Vitamin D "
            "supplementation was associated with a 10% reduction in the risk "
            "of hip fractures (RR 0.90, 95% CI 0.83-0.98). CONCLUSIONS: "
            "Vitamin D supplementation may reduce the risk of hip fractures in "
            "older adults, but the benefit is modest."
        ),
        "summary": (
            "Vitamin D supplementation is associated with a modest reduction "
            "in hip fracture risk in older adults, based on meta-analysis of "
            "11 trials."
        )
    }
]


class LLMPipeline:
    """
    Pipeline B per generazione astrattiva con LLM tramite Ollama.

    Args:
        model_name:          Nome del modello Ollama (es. 'mixtral', 'mistral', 'llama3')
        prompting_strategy:  'zero-shot', 'few-shot', 'cot'
        temperature:         Temperatura di sampling (bassa per ridurre allucinazioni)
        max_new_tokens:      Numero massimo di token generati
        ollama_host:         URL base dell'istanza Ollama (default: http://localhost:11434)
    """

    def __init__(
        self,
        model_name: str,
        prompting_strategy: str = "few-shot",
        temperature: float = 0.1,
        max_new_tokens: int = 4096,
        ollama_host: str = "http://localhost:11434",
        # Mantenuti per compatibilità con il codice esistente, non usati con Ollama
        use_4bit: bool = False,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.ollama_host = ollama_host.rstrip("/")
        self.api_url = f"{self.ollama_host}/api/generate"

        logger.info(f"Connessione a Ollama: {self.ollama_host} — modello: {model_name}")
        self._check_ollama()

    def _check_ollama(self):
        """Verifica che Ollama sia raggiungibile e che il modello sia disponibile."""
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            # Controlla sia il nome esatto che il nome senza tag (es. "mixtral" vs "mixtral:latest")
            names_bare = [n.split(":")[0] for n in available]
            if self.model_name not in available and self.model_name not in names_bare:
                logger.warning(
                    f"Modello '{self.model_name}' non trovato in Ollama. "
                    f"Disponibili: {available}. "
                    f"Esegui: ollama pull {self.model_name}"
                )
            else:
                logger.info(f"Modello '{self.model_name}' disponibile su Ollama.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama non raggiungibile su {self.ollama_host}. "
                "Assicurati che il servizio sia avviato con: ollama serve"
            )

    def _build_prompt(self, article: str) -> str:
        """Costruisce il prompt in base alla strategia selezionata."""
        if self.prompting_strategy == "zero-shot":
            return self._zero_shot_prompt(article)
        elif self.prompting_strategy == "few-shot":
            return self._few_shot_prompt(article)
        elif self.prompting_strategy == "cot":
            return self._cot_prompt(article)
        else:
            raise ValueError(f"Strategia non supportata: {self.prompting_strategy}")

    def _zero_shot_prompt(self, article: str) -> str:
        return (
            f"### Instruction:\n"
            f"Read the following medical article and write a concise, abstractive summary "
            f"in one or two sentences. Capture the main findings and conclusions.\n\n"
            f"### Article:\n{article}\n\n"
            f"### Summary:\n"
        )

    def _few_shot_prompt(self, article: str) -> str:
        few_shot_text = ""
        for ex in FEW_SHOT_EXAMPLES:
            few_shot_text += (
                f"### Article:\n{ex['article']}\n\n"
                f"### Summary:\n{ex['summary']}\n\n"
            )
        return (
            f"### Instruction:\n"
            f"Read the following medical article and write a concise, abstractive summary "
            f"in one or two sentences. Capture the main findings and conclusions.\n\n"
            f"Here are some examples:\n\n"
            f"{few_shot_text}"
            f"### Article:\n{article}\n\n"
            f"### Summary:\n"
        )

    def _cot_prompt(self, article: str) -> str:
        return (
            f"### Instruction:\n"
            f"Read the following medical article and provide a step-by-step reasoning "
            f"about the most important points, then produce a concise abstractive summary.\n\n"
            f"### Article:\n{article}\n\n"
            f"### Chain of Thought:\n"
            f"1. The main objective of the study is ...\n"
            f"2. The key findings are ...\n"
            f"3. The conclusion is ...\n"
            f"### Summary:\n"
        )

    def generate_summary(self, article: str) -> str:
        """Genera un riassunto per un singolo articolo tramite Ollama."""
        # Gestisce NaN (float) o qualsiasi valore non-stringa
        if not isinstance(article, str) or not article.strip():
            return ""

        prompt = self._build_prompt(article)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "top_p": 0.95,
                "repeat_penalty": 1.15,
            },
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=120)
            resp.raise_for_status()
            summary = resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            logger.error("Timeout nella richiesta a Ollama.")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore nella richiesta a Ollama: {e}")
            return ""

        # Se la strategia è CoT, estraiamo la parte dopo "### Summary:"
        if self.prompting_strategy == "cot":
            for marker in ["### Summary:", "Summary:", "In summary,"]:
                if marker in summary:
                    summary = summary.split(marker)[-1].strip()
                    break
            else:
                sentences = [s.strip() for s in summary.split(".") if s.strip()]
                summary = ". ".join(sentences[-2:]) + "."

        return summary.strip()

    def run(self, text: str) -> str:
        """Interfaccia pubblica per un singolo testo."""
        return self.generate_summary(text)

    def run_batch(self, texts: list[str]) -> list[str]:
        """Esegue su una lista di testi (sequenziale)."""
        summaries = []
        total = len(texts)
        for i, article in enumerate(texts):
            if i % 50 == 0 and i > 0:
                logger.info(f"  LLM: {i}/{total} articoli processati")
            if not isinstance(article, str):
                logger.warning(f"  Riga {i} ignorata: tipo non valido ({type(article).__name__}: {article!r})")
                summaries.append("")
                continue
            summaries.append(self.generate_summary(article))
        return summaries