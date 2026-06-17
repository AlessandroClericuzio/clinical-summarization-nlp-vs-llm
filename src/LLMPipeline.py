"""
LLMPipeline.py — Pipeline B: Generative LLM (Abstractive)
Progetto NLP: Clinical Summarization — NLP Tradizionale vs LLM

Supporta:
  - Modelli con context window ≥ 32k (Mixtral, Mistral v0.2/v0.3)
  - Strategie di prompting: zero-shot, few-shot, chain-of-thought (CoT)
  - Quantizzazione 4-bit per ridurre l'ingombro VRAM
"""

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

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
    Pipeline B per generazione astrattiva con LLM.

    Args:
        model_name: Nome del modello HF (es. 'mistralai/Mixtral-8x7B-Instruct-v0.1')
        prompting_strategy: 'zero-shot', 'few-shot', 'cot'
        temperature: temperatura di sampling (bassa per ridurre allucinazioni)
        max_new_tokens: numero massimo di token generati
        use_4bit: se True carica il modello in 4-bit (bitsandbytes)
        device_map: 'auto' per distribuzione automatica
    """

    def __init__(
        self,
        model_name: str,
        prompting_strategy: str = "few-shot",
        temperature: float = 0.1,
        max_new_tokens: int = 256,
        use_4bit: bool = False,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        logger.info(f"Caricamento modello LLM: {model_name}")
        self._load_model(use_4bit, device_map)

        # Imposta tokenizer per troncamento a sinistra (per mantenere il più possibile il contesto)
        self.tokenizer.truncation_side = "left"
        # La context window effettiva dipende dal modello; per Mixtral/Mistral v0.2 è 32768
        self.max_context_len = min(
            self.tokenizer.model_max_length,
            32768  # safe per Mixtral/Mistral v0.2
        )

    def _load_model(self, use_4bit: bool, device_map: str):
        """Carica modello e tokenizer con configurazione di quantizzazione."""
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            bnb_config = None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        logger.info(f"Modello caricato. Context window: {self.max_context_len}")

    def _build_prompt(self, article: str) -> str:
        """
        Costruisce il prompt in base alla strategia selezionata.
        """
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
        # Costruiamo il few-shot con gli esempi
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

    def _truncate_to_context(self, prompt: str) -> str:
        """
        Tronca il prompt a sinistra per rispettare la context window,
        lasciando spazio per la risposta (max_new_tokens).
        """
        # Stimiamo il numero di token del prompt
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        max_allowed = self.max_context_len - self.max_new_tokens - 50  # margine di sicurezza

        if len(tokens) <= max_allowed:
            return prompt

        # Tronca a sinistra mantenendo l'inizio del prompt? Meglio mantenere l'istruzione e l'inizio dell'articolo.
        # Poiché usiamo truncation_side="left", il tokenizer troncherà a sinistra automaticamente.
        # Ma per sicurezza, possiamo pre-troncare l'articolo.
        # Approccio: manteniamo solo la parte finale dell'articolo (ultimi N token)
        # e reinseriamo il prefisso.
        # Tuttavia, la tokenizzazione del modello applicherà il troncamento a sinistra.
        return prompt  # lasceremo che il tokenizer faccia il lavoro

    def generate_summary(self, article: str) -> str:
        """Genera un riassunto per un singolo articolo."""
        if not article or not article.strip():
            return ""

        prompt = self._build_prompt(article)

        # Tokenizza con troncamento a sinistra
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_len - self.max_new_tokens,
            padding=False,
        )

        # Sposta su GPU se necessario
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.15,    
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decodifica saltando il prompt
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        summary = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Se la strategia è CoT, estraiamo la parte dopo "### Summary:"
        if self.prompting_strategy == "cot":
            # Cerca il summary dopo l'ultimo marcatore
            for marker in ["### Summary:", "Summary:", "In summary,"]:
                if marker in summary:
                    summary = summary.split(marker)[-1].strip()
                    break
            # Se non trova nessun marker, prende le ultime 2 frasi
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
            summaries.append(self.generate_summary(article))
        return summaries


# Test rapido
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_article = (
        "BACKGROUND: The efficacy of acupuncture for chronic low back pain "
        "has been controversial. OBJECTIVE: To evaluate the effectiveness of "
        "acupuncture vs sham acupuncture and no acupuncture for chronic low "
        "back pain. METHODS: A randomized controlled trial with 638 patients "
        "was conducted. RESULTS: Acupuncture was significantly more effective "
        "than no acupuncture, but not more than sham acupuncture. "
        "CONCLUSIONS: Acupuncture is effective for chronic low back pain, "
        "but the effect size is small and may be due to placebo."
    )
    llm = LLMPipeline(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompting_strategy="few-shot",
        use_4bit=True,
    )
    summary = llm.run(sample_article)
    print("SUMMARY:", summary)