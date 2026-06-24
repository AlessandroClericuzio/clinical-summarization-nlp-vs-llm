# clinical-summarization-nlp-vs-llm
A comparative benchmarking study evaluating traditional NLP pipelines (extractive parsing &amp; NER) against Large Language Models (generative prompting) for medical intent summarization and clinical information extraction using the NIH MeQSum dataset.

Per installare le librerie necessarie

sudo apt-get update && sudo apt-get install -y zstd

curl -fsSL https://ollama.com/install.sh | sh

ollama serve

Per lanciarlo
python main.py --dataset_name ccdv/pubmed-summarization --sample_size 500 --llm_model llama3.1:8b --prompting zero-shot