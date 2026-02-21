# SRT English → Farsi AI Translator

A robust CLI tool that translates `.srt` subtitle files from English to Persian (Farsi) using an AI model.

## Features
- Parses and preserves SRT structure (index + timestamp)
- Translates in configurable batches
- Retries failed batches automatically
- Supports:
  - OpenAI-compatible Chat Completions APIs
  - Ollama local server (`/api/chat`)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
### 1) OpenAI-compatible API
```bash
export OPENAI_API_KEY="YOUR_KEY"
python translate_srt.py input.srt output.fa.srt \
  --provider openai \
  --model gpt-4o-mini \
  --base-url https://api.openai.com/v1
```

### 2) Ollama local model
```bash
# example model: qwen2.5:7b
python translate_srt.py input.srt output.fa.srt \
  --provider ollama \
  --model qwen2.5:7b \
  --base-url http://localhost:11434
```

## Advanced options
- `--batch-size 20` subtitles per request
- `--retries 3`
- `--backoff 1.5`
- `--temperature 0.2`
- `--timeout 120`

## Notes
- Keep your source `.srt` UTF-8 encoded.
- Output remains valid `.srt` with translated text.
- If your backend is OpenAI-compatible, keep `--provider openai` and point `--base-url` accordingly.
