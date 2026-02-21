#!/usr/bin/env python3
"""Translate .srt subtitle files from English to Farsi with an AI model.

Features:
- Robust SRT parsing/writing
- Batch translation with retries
- OpenAI-compatible API support (OpenAI, local proxies, many hosted gateways)
- Optional Ollama support
- Preserves subtitle numbering and timestamps
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

@dataclass
class SRTEntry:
    index: int
    timestamp: str
    text_lines: List[str]


SRT_BLOCK_RE = re.compile(
    r"(?ms)^\s*(\d+)\s*\n"
    r"([0-9:,]+\s*-->\s*[0-9:,]+)\s*\n"
    r"(.*?)(?=\n\s*\n|\Z)"
)


def parse_srt(content: str) -> List[SRTEntry]:
    entries: List[SRTEntry] = []
    for match in SRT_BLOCK_RE.finditer(content.strip() + "\n"):
        idx = int(match.group(1))
        ts = match.group(2).strip()
        text = [line.rstrip() for line in match.group(3).splitlines()]
        entries.append(SRTEntry(index=idx, timestamp=ts, text_lines=text))

    if not entries:
        raise ValueError("No valid subtitle blocks found. Is this a valid .srt file?")
    return entries


def write_srt(entries: Iterable[SRTEntry]) -> str:
    parts: List[str] = []
    for e in entries:
        parts.append(str(e.index))
        parts.append(e.timestamp)
        parts.extend(e.text_lines)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def chunked(items: List[SRTEntry], size: int) -> Iterable[List[SRTEntry]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def batch_payload(entries: List[SRTEntry]) -> str:
    payload = []
    for e in entries:
        payload.append({"id": e.index, "text": "\n".join(e.text_lines)})
    return json.dumps(payload, ensure_ascii=False)


def parse_batch_response(raw: str) -> dict[int, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON output: {raw[:300]}") from exc

    if not isinstance(data, list):
        raise ValueError("Model response must be a JSON array.")

    out: dict[int, str] = {}
    for obj in data:
        if not isinstance(obj, dict):
            raise ValueError("Each response item must be an object.")
        if "id" not in obj or "text" not in obj:
            raise ValueError("Each response item must include 'id' and 'text'.")
        out[int(obj["id"])] = str(obj["text"])
    return out


def openai_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    import requests
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def ollama_chat(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }
    import requests
    resp = requests.post(url, json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def translate_entries(
    entries: List[SRTEntry],
    translate_fn: Callable[[str, str], str],
    batch_size: int,
    retries: int,
    backoff_s: float,
) -> List[SRTEntry]:
    result: List[SRTEntry] = []

    system_prompt = (
        "You are an expert subtitle translator. Translate English subtitles to Persian (Farsi). "
        "Keep tone natural and concise for spoken dialogue. "
        "Output ONLY valid JSON array. Do not add markdown fences or explanations."
    )

    for batch in chunked(entries, batch_size):
        user_prompt = (
            "Translate each 'text' value from English to Farsi. Preserve line breaks where helpful. "
            "Do not censor. Keep punctuation natural in Farsi.\n\n"
            "Return this exact JSON shape:\n"
            "[{\"id\": 1, \"text\": \"...\"}]\n\n"
            "Input JSON:\n"
            f"{batch_payload(batch)}"
        )

        last_err: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                raw = translate_fn(system_prompt, user_prompt)
                mapped = parse_batch_response(raw)

                for e in batch:
                    if e.index not in mapped:
                        raise ValueError(f"Missing id {e.index} in model response.")
                    translated_lines = mapped[e.index].splitlines() or [""]
                    result.append(
                        SRTEntry(
                            index=e.index,
                            timestamp=e.timestamp,
                            text_lines=translated_lines,
                        )
                    )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt == retries:
                    raise RuntimeError(
                        f"Failed translating batch starting at subtitle #{batch[0].index}: {exc}"
                    ) from exc
                time.sleep(backoff_s * attempt)
        if last_err is not None and len(result) < entries.index(batch[0]):
            raise RuntimeError(f"Unexpected translation state: {last_err}")

    return result


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Translate .srt subtitles from English to Farsi.")
    p.add_argument("input", type=Path, help="Path to source .srt file")
    p.add_argument("output", type=Path, help="Path for translated .srt file")

    p.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="AI backend provider",
    )
    p.add_argument(
        "--model",
        default=os.getenv("SRT_MODEL", "gpt-4o-mini"),
        help="Model name",
    )
    p.add_argument(
        "--base-url",
        default=os.getenv("SRT_BASE_URL", "https://api.openai.com/v1"),
        help="Provider base URL (OpenAI-compatible for --provider openai)",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (for OpenAI-compatible providers)",
    )

    p.add_argument("--batch-size", type=int, default=20, help="Subtitles per translation call")
    p.add_argument("--retries", type=int, default=3, help="Retries per failed batch")
    p.add_argument("--backoff", type=float, default=1.5, help="Retry backoff seconds")
    p.add_argument("--temperature", type=float, default=0.2, help="Model temperature")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds")
    return p


def main() -> int:
    args = build_cli().parse_args()

    if args.provider == "openai" and not args.api_key:
        print(
            "Error: API key required for provider=openai. Set OPENAI_API_KEY or --api-key.",
            file=sys.stderr,
        )
        return 2

    source = args.input.read_text(encoding="utf-8")
    entries = parse_srt(source)

    if args.provider == "openai":
        def call(system_prompt: str, user_prompt: str) -> str:
            return openai_chat(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                timeout=args.timeout,
            )
    else:
        def call(system_prompt: str, user_prompt: str) -> str:
            return ollama_chat(
                base_url=args.base_url,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                timeout=args.timeout,
            )

    translated = translate_entries(
        entries=entries,
        translate_fn=call,
        batch_size=args.batch_size,
        retries=args.retries,
        backoff_s=args.backoff,
    )

    args.output.write_text(write_srt(translated), encoding="utf-8")
    print(f"Translated {len(entries)} subtitles -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
