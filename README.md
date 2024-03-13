# Haystack - Cloudflare Workers AI + LlamaGuard

This is an exploration of the deepset orchestration framework Haystack using Cloudflare Workers AI.

It is VERY MUCH A WORK IN PROGRESS.

Tread lightly.

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