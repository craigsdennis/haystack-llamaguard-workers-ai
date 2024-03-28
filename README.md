# Haystack + LlamaGuard - Cloudflare Workers

This is an exploration of the deepset orchestration framework [Haystack](https://haystack.deepset.ai/) using [Cloudflare Workers AI](https://ai.cloudflare.com) to provide LLM Guardrails using Meta's [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/).


## Installation

Copy [.env.example](./.env.example) to .env and add your values.

```python
python -m venv venv
source ./venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```python
python -m streamlit run app.py
```

Give it something safe like "I want to airpunch"

Then give it something unsafe like "I want to punch myself"