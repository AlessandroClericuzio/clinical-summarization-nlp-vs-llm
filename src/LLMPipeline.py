"""
LLMPipeline.py — Pipeline B: Generative LLM (Abstractive)
Progetto NLP: Clinical Summarization — NLP Tradizionale vs LLM

Supporta:
  - Qualsiasi modello disponibile su Ollama (es. mixtral, mistral, llama3, ecc.)
  - Strategie di prompting: zero-shot, one-shot, few-shot, chain-of-thought (CoT)
  - Backend: Ollama (HTTP API locale)

Modifiche rispetto alla versione originale:
  - [Bug fix] generate_summary(): rimossa chiamata a _call_ollama() inesistente
    che causava doppia esecuzione e risultati inconsistenti.
  - [Bug fix] Aggiunta strategia "one-shot" per allineamento con DEFAULT_PROMPTING di main.py.
  - [Miglioramento] Post-processing CoT reso più robusto con regex.
  - [Miglioramento] Logging del prompt usato per debug.
  - [Miglioramento] Timeout aumentato e configurabile.
"""

import logging
import re
import requests
from typing import Optional

logger = logging.getLogger(__name__)

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

# Sottoinsieme di un esempio per one-shot
ONE_SHOT_EXAMPLE = FEW_SHOT_EXAMPLES[0]


class LLMPipeline:
    """
    Pipeline B per generazione astrattiva con LLM tramite Ollama.

    Args:
        model_name:          Nome del modello Ollama (es. 'mixtral', 'mistral', 'llama3')
        prompting_strategy:  'zero-shot', 'one-shot', 'few-shot', 'cot'
        temperature:         Temperatura di sampling (bassa per ridurre allucinazioni)
        max_new_tokens:      Numero massimo di token generati
        ollama_host:         URL base dell'istanza Ollama (default: http://localhost:11434)
        request_timeout:     Timeout in secondi per la richiesta HTTP (default: 180)
    """

    VALID_STRATEGIES = {"zero-shot", "one-shot", "few-shot", "cot"}

    def __init__(
        self,
        model_name: str = "gemma4:26b",
        prompting_strategy: str = "few-shot",
        temperature: float = 0.1,
        max_new_tokens: int = 4096,
        ollama_host: str = "http://localhost:11434",
        request_timeout: int = 180,
        # Mantenuti per compatibilità con il codice esistente, non usati con Ollama
        use_4bit: bool = False,
        device_map: str = "auto",
    ):
        if prompting_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Strategia non supportata: '{prompting_strategy}'. "
                f"Scegli tra: {sorted(self.VALID_STRATEGIES)}"
            )

        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.ollama_host = ollama_host.rstrip("/")
        self.api_url = f"{self.ollama_host}/api/generate"
        self.request_timeout = request_timeout

        logger.info(
            f"Connessione a Ollama: {self.ollama_host} — "
            f"modello: {model_name} — "
            f"strategia: {prompting_strategy}"
        )
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

    # ─────────────────────────────────────────────
    # Costruzione prompt
    # ─────────────────────────────────────────────

    def _build_prompt(self, article: str) -> str:
        """Costruisce il prompt in base alla strategia selezionata."""
        dispatch = {
            "zero-shot": self._zero_shot_prompt,
            "one-shot":  self._one_shot_prompt,
            "few-shot":  self._few_shot_prompt,
            "cot":       self._cot_prompt,
        }
        return dispatch[self.prompting_strategy](article)

    def _zero_shot_prompt(self, article: str) -> str:
        return (
            "### Instruction:\n"
            "Read the following medical article and write a concise, abstractive summary "
            "in one or two sentences. Capture the main findings and conclusions. "
            "Output only the summary, no preamble.\n\n"
            f"### Article:\n{article}\n\n"
            "### Summary:\n"
        )

    def _one_shot_prompt(self, article: str) -> str:
        ex = ONE_SHOT_EXAMPLE
        return (
            "### Instruction:\n"
            "Read the following medical article and write a concise, abstractive summary "
            "in one or two sentences. Capture the main findings and conclusions. "
            "Output only the summary, no preamble.\n\n"
            "Here is an example:\n\n"
            f"### Article:\n{ex['article']}\n\n"
            f"### Summary:\n{ex['summary']}\n\n"
            f"### Article:\n{article}\n\n"
            "### Summary:\n"
        )

    def _few_shot_prompt(self, article: str) -> str:
        few_shot_text = ""
        for ex in FEW_SHOT_EXAMPLES:
            few_shot_text += (
                f"### Article:\n{ex['article']}\n\n"
                f"### Summary:\n{ex['summary']}\n\n"
            )
        return (
            "### Instruction:\n"
            "Read the following medical article and write a concise, abstractive summary "
            "in one or two sentences. Capture the main findings and conclusions. "
            "Output only the summary, no preamble.\n\n"
            "Here are some examples:\n\n"
            f"{few_shot_text}"
            f"### Article:\n{article}\n\n"
            "### Summary:\n"
        )

    def _cot_prompt(self, article: str) -> str:
        return (
            "### Instruction:\n"
            "Read the following medical article. First reason step by step about the "
            "key points, then write a concise abstractive summary in one or two sentences.\n\n"
            f"### Article:\n{article}\n\n"
            "### Chain of Thought:\n"
            "1. The main objective of the study is ...\n"
            "2. The key findings are ...\n"
            "3. The conclusion is ...\n"
            "### Summary:\n"
        )

    # ─────────────────────────────────────────────
    # Generazione
    # ─────────────────────────────────────────────

    def generate_summary(self, article: str) -> str:
        """
        Genera un riassunto per un singolo articolo tramite Ollama.

        Fix rispetto all'originale:
          - Rimossa chiamata a self._call_ollama() (metodo inesistente) che
            causava un AttributeError silenzioso e una doppia esecuzione della POST.
          - Il payload viene costruito una sola volta e inviato in un unico blocco try/except.
        """
        if not isinstance(article, str) or not article.strip():
            return ""

        prompt = self._build_prompt(article)
        logger.debug(f"Prompt ({self.prompting_strategy}):\n{prompt[:200]}…")

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
            resp = requests.post(self.api_url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            summary = resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            logger.error(
                f"Timeout ({self.request_timeout}s) nella richiesta a Ollama. "
                "Considera di aumentare request_timeout o ridurre max_new_tokens."
            )
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Errore nella richiesta a Ollama: {e}")
            return ""

        return self._postprocess(summary)

    def _postprocess(self, summary: str) -> str:
        """
        Post-processing dell'output del modello.

        Per CoT estrae la parte dopo l'ultimo marcatore "### Summary:".
        Per tutte le strategie rimuove eventuali prefissi residui del prompt.
        """
        if self.prompting_strategy == "cot":
            # Cerca il marcatore ### Summary: con regex case-insensitive
            match = re.search(r"###\s*Summary\s*:(.*)", summary, re.IGNORECASE | re.DOTALL)
            if match:
                summary = match.group(1).strip()
            else:
                # Fallback: cerca "In summary," o "Summary:"
                for marker in ["In summary,", "Summary:", "In conclusion,"]:
                    if marker.lower() in summary.lower():
                        idx = summary.lower().index(marker.lower()) + len(marker)
                        summary = summary[idx:].strip()
                        break
                else:
                    # Ultimo fallback: ultime 2 frasi
                    sentences = [s.strip() for s in summary.split(".") if s.strip()]
                    summary = ". ".join(sentences[-2:]) + ("." if sentences else "")

        # Rimuovi eventuali prefissi del prompt rimasti nell'output
        for prefix in ["### Summary:", "Summary:", "Abstract:"]:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()

        return summary.strip()

    # ─────────────────────────────────────────────
    # Interfaccia pubblica
    # ─────────────────────────────────────────────

    def run(self, text: str) -> str:
        return self.generate_summary(text)

    def run_batch(self, texts: list[str]) -> list[str]:
        summaries = []
        total = len(texts)
        for i, article in enumerate(texts):
            if i % 50 == 0 and i > 0:
                logger.info(f"  LLM: {i}/{total} articoli processati")
            if not isinstance(article, str):
                logger.warning(
                    f"  Riga {i} ignorata: tipo non valido "
                    f"({type(article).__name__}: {article!r})"
                )
                summaries.append("")
                continue
            summaries.append(self.generate_summary(article))
        return summaries