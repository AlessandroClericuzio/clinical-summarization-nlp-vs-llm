import logging
from typing import Optional
import requests
import json

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

class LLMPipeline:
    def __init__(
        self,
        model_name: str = "gemma4:26b",
        prompting_strategy: str = "few-shot",
        temperature: float = 0.1,
        max_new_tokens: int = 2048,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self._generate_url = f"{self.ollama_base_url}/api/chat"

        logger.info(f"LLMPipeline initialized with model: {model_name} via Ollama")

    def _build_prompt(self, article: str) -> str:
        if self.prompting_strategy == "zero-shot":
            return self._zero_shot_prompt(article)
        elif self.prompting_strategy == "few-shot":
            return self._few_shot_prompt(article)
        elif self.prompting_strategy == "cot":
            return self._cot_prompt(article)
        else:
            raise ValueError(f"Unsupported strategy: {self.prompting_strategy}")

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

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
                "top_p": 0.95,
                "repeat_penalty": 1.15,
            }
        }
        response = requests.post(self._generate_url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    def generate_summary(self, article: str) -> str:
        if not isinstance(article, str) or not article.strip():
            return ""

        prompt = self._build_prompt(article)
        summary = self._call_ollama(prompt)

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
        return self.generate_summary(text)

    def run_batch(self, texts: list[str]) -> list[str]:
        summaries = []
        total = len(texts)
        for i, article in enumerate(texts):
            if i % 50 == 0 and i > 0:
                logger.info(f"  LLM: {i}/{total} articoli processati")
            summaries.append(self.generate_summary(article))
        return summaries


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
        model_name="qwen3.5:27b",
        prompting_strategy="few-shot",
    )
    summary = llm.run(sample_article)
    print("SUMMARY:", summary)