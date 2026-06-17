# clinical-summarization-nlp-vs-llm
A comparative benchmarking study evaluating traditional NLP pipelines (extractive parsing &amp; NER) against Large Language Models (generative prompting) for medical intent summarization and clinical information extraction using the NIH MeQSum dataset.


Per lanciarlo
python main.py --dataset_name ccdv/pubmed-summarization --sample_size 500 --llm_model mistralai/Mixtral-8x7B-Instruct-v0.1 --prompting few-shot
