import logging
import re
import requests
from typing import Optional

logger = logging.getLogger(__name__)

def _load_real_examples(n: int = 2) -> list[dict]:
    # Load real reference examples from the PubMed dataset using fixed indices
    from datasets import load_dataset
    ds = load_dataset("ccdv/pubmed-summarization", split="train")
    fixed_indices = [10000, 10001, 10002][:n]
    return [
        {"article": ds[i]["article"], "abstract": ds[i]["abstract"]}
        for i in fixed_indices
    ]

_REAL_EXAMPLES = _load_real_examples(n=2)
ONE_SHOT_EXAMPLE  = _REAL_EXAMPLES[:1]
FEW_SHOT_EXAMPLES = _REAL_EXAMPLES[:2]

class LLMPipeline:

    VALID_STRATEGIES = {"zero-shot", "one-shot", "few-shot", "cot"}
    _TASK_INSTRUCTION = (
        "You are a biomedical expert specialized in summarizing scientific articles.\n"
        "Your task is to produce a concise, faithful, and abstractive summary "
        "of a biomedical research article.\n\n"
        "Rules:\n"
        "- Write 2 to 4 sentences.\n"
        "- Cover: main objective, key findings, and conclusion.\n"
        "- Use only information explicitly present in the article.\n"
        "- Do not add interpretations, opinions, or external knowledge.\n"
        "- Output only the summary, with no preamble or label.\n"
    )

    def __init__(
        self,
        model_name: str = "gemma4:26b",
        prompting_strategy: str = "few-shot",
        temperature: float = 0.1,
        max_new_tokens: int = 4096,
        ollama_host: str = "http://localhost:11434",
        request_timeout: int = 180,
        use_4bit: bool = False,
        device_map: str = "auto",
    ):
        if prompting_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unsupported strategy: '{prompting_strategy}'. "
                f"Choose from: {sorted(self.VALID_STRATEGIES)}"
            )

        self.model_name = model_name
        self.prompting_strategy = prompting_strategy
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.ollama_host = ollama_host.rstrip("/")
        self.api_url = f"{self.ollama_host}/api/generate"
        self.request_timeout = request_timeout

        logger.info(
            f"Connecting to Ollama: {self.ollama_host} - "
            f"model: {model_name} - "
            f"strategy: {prompting_strategy}"
        )
        self._check_ollama()

    def _check_ollama(self):
        # Verify Ollama service availability and check if the requested model is accessible
        try:
            resp = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]
            names_bare = [n.split(":")[0] for n in available]
            if self.model_name not in available and self.model_name not in names_bare:
                logger.warning(
                    f"Model '{self.model_name}' not found in Ollama. "
                    f"Available: {available}. "
                    f"Run: ollama pull {self.model_name}"
                )
            else:
                logger.info(f"Model '{self.model_name}' is available on Ollama.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama is unreachable at {self.ollama_host}. "
                "Ensure the service is running via: ollama serve"
            )

    def _build_prompt(self, article: str) -> str:
        # Dispatches prompt building based on the selected strategy
        dispatch = {
            "zero-shot": self._zero_shot_prompt,
            "one-shot":  self._one_shot_prompt,
            "few-shot":  self._few_shot_prompt,
            "cot":       self._cot_prompt,
        }
        return dispatch[self.prompting_strategy](article)

    def _zero_shot_prompt(self, article: str) -> str:
        return (
            f"### Task:\n{self._TASK_INSTRUCTION}\n"
            f"### Article:\n{article}\n\n"
            "### Summary:\n"
        )

    def _one_shot_prompt(self, article: str) -> str:
        ex = ONE_SHOT_EXAMPLE[0]
        return (
            f"### Task:\n{self._TASK_INSTRUCTION}\n"
            "### Reference Example:\n\n"
            f"Article:\n{ex['article']}\n\n"
            f"Summary:\n{ex['abstract']}\n\n"
            "---\n\n"
            f"### Article:\n{article}\n\n"
            "### Summary:\n"
        )

    def _few_shot_prompt(self, article: str) -> str:
        references = ""
        for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
            references += (
                f"Reference {i}:\n"
                f"Article:\n{ex['article']}\n\n"
                f"Summary:\n{ex['abstract']}\n\n"
                "---\n\n"
            )
        return (
            f"### Task:\n{self._TASK_INSTRUCTION}\n"
            f"### Reference Examples:\n\n{references}"
            f"### Article:\n{article}\n\n"
            "### Summary:\n"
        )

    def _cot_prompt(self, article: str) -> str:
        return (
            f"### Task:\n{self._TASK_INSTRUCTION}\n"
            "Before writing the summary, reason step by step:\n\n"
            f"### Article:\n{article}\n\n"
            "### Reasoning:\n"
            "1. The main objective of this study is: ...\n"
            "2. The methodology used is: ...\n"
            "3. The key findings are: ...\n"
            "4. The main conclusion is: ...\n\n"
            "### Summary:\n"
        )

    def generate_summary(self, article: str) -> str:
        if not isinstance(article, str) or not article.strip():
            return ""

        prompt = self._build_prompt(article)
        estimated_tokens = len(prompt.split()) * 1.3
        if estimated_tokens > 30000:
            logger.warning(f"Prompt estimated at {estimated_tokens:.0f} tokens - close to context window limit")
        logger.debug(f"Prompt ({self.prompting_strategy}):\n{prompt[:200]}...")

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
                f"Timeout ({self.request_timeout}s) during Ollama request. "
                "Consider increasing request_timeout or decreasing max_new_tokens."
            )
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during Ollama request: {e}")
            return ""

        return self._postprocess(summary)

    def _postprocess(self, summary: str) -> str:
        # Extract the final section after the structural summary marker for CoT strategies
        if self.prompting_strategy == "cot":
            match = re.search(r"###\s*Summary\s*:(.*)", summary, re.IGNORECASE | re.DOTALL)
            if match:
                summary = match.group(1).strip()
            else:
                for marker in ["In summary,", "Summary:", "In conclusion,"]:
                    if marker.lower() in summary.lower():
                        idx = summary.lower().index(marker.lower()) + len(marker)
                        summary = summary[idx:].strip()
                        break
                else:
                    sentences = [s.strip() for s in summary.split(".") if s.strip()]
                    summary = ". ".join(sentences[-2:]) + ("." if sentences else "")

        # Strip remaining prompt structure headers if leaky model generation occurred
        for prefix in ["### Summary:", "Summary:", "Abstract:"]:
            if summary.startswith(prefix):
                summary = summary[len(prefix):].strip()

        return summary.strip()

    def run(self, text: str) -> str:
        return self.generate_summary(text)

    def run_batch(self, texts: list[str]) -> list[str]:
        summaries = []
        total = len(texts)
        for i, article in enumerate(texts):
            if i % 50 == 0 and i > 0:
                logger.info(f"LLM: {i}/{total} articles processed")
            if not isinstance(article, str):
                logger.warning(
                    f"Row {i} ignored: invalid type "
                    f"({type(article).__name__}: {article!r})"
                )
                summaries.append("")
                continue
            summaries.append(self.generate_summary(article))
        return summaries