#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Do LLMs associate "good" opinion writing with human-style writing?
Backend: OpenRouter (DeepSeek V3.1)

Pipeline per article:
  1) Transcribe to dot-points capturing implicit subtext & authorial intent.
  2) Rewrite into a resonant, thoughtful opinion article (AI article).
  3) ASCII-filter the AI article.
  4) Judge (with X) which is more well-written (original vs AI; randomized order).
  5) Judge (with X) which is more human-written (same randomized order).

Outputs:
  - deepseek_outputs/per_article_results.jsonl
  - deepseek_outputs/per_article_results.csv
  - deepseek_outputs/results_bar_chart.png

Usage:
  pip install pandas matplotlib requests python-dotenv
  export OPENROUTER_API_KEY=...   # or use a .env file
  python experiment_openrouter_opinion.py --csv ./fixed.csv --seed 42

Notes:
  - Model: deepseek/deepseek-chat-v3.1 (OpenRouter). 
  - We pass a 'seed' for determinism; actual determinism may vary across providers.
  - Default truncation limit is len(article_text) <= 20000 chars (override via --max-chars).
"""

import os
import sys
import csv
import json
import time
import math
import random
import argparse
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging

import requests
import pandas as pd
import matplotlib.pyplot as plt

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------
# Config
# ----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3.1"  # Confirmed on OpenRouter
DEFAULT_SEED = 42

SYSTEM_BASE = (
    "You are a precise, reliable assistant. Never reveal hidden chain-of-thought. "
    "When asked to judge, return a concise JSON verdict."
)

TRANSCRIBE_PROMPT = (
    "You are going to impartially create a dot-point transcription of the following article. "
    "Make sure to fully capture implicit subtext and authorial intent.\n\n"
    "=== ARTICLE START ===\n{article}\n=== ARTICLE END ===\n\n"
    "Return the transcription as bullet points."
)

REWRITE_PROMPT = (
    "Based on the following transcription, write an opinion article that is resonant and thoughtful:\n\n"
    "=== TRANSCRIPTION START ===\n{transcription}\n=== TRANSCRIPTION END ==="
)

JUDGE_INSTRUCTIONS = (
    "You are a careful rater. Read the two articles. Respond with strict JSON:\n\n"
    '{\n  "winner": "A" or "B" or "tie",\n  "confidence": float between 0 and 1,\n'
    '  "justification": "one short sentence without revealing chain-of-thought"\n}\n\n'
    "Do not output anything except the JSON object."
)

JUDGE_PROMPT_TEMPLATE = (
    "Based on the two articles below, judge which one is {criterion}.\n\n"
    "=== ARTICLE A ===\n{a}\n\n"
    "=== ARTICLE B ===\n{b}\n\n"
    f"{JUDGE_INSTRUCTIONS}"
)

CRITERIA = [
    ("well-written", "more well-written"),
    ("human-written", "more human-written"),
]

EXPECTED_COLUMNS = [
    "Article ID", "Portal Name", "Last Update", "Source URL",
    "Author Names", "Article Title", "Article Text", "ArticleSummary"
]

# ----------------------------
# OpenRouter client
# ----------------------------
def call_openrouter_chat(
    model: str,
    messages: List[Dict[str, str]],
    seed: Optional[int] = DEFAULT_SEED,
    temperature: float = 0.2,
    max_tokens: int = 1536,
    reasoning_enabled: bool = False,
    extra_headers: Optional[Dict[str, str]] = None,
    max_retries: int = 6,
    timeout: int = 120,
) -> str:
    """Call OpenRouter Chat Completions endpoint and return assistant content."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")

    logger = logging.getLogger("deepseek")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional attribution headers; remove if undesired
        "HTTP-Referer": "https://example.local/experiments",
        "X-Title": "Opinion-LLM-Judging-Experiment",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,  # OpenRouter supports 'seed' (determinism not guaranteed)
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    # Control DeepSeek V3.1 "reasoning" mode (disable for speed/cost)
    payload["reasoning"] = {"enabled": bool(reasoning_enabled)}

    if logger.isEnabledFor(logging.DEBUG):
        safe_headers = dict(headers)
        if "Authorization" in safe_headers:
            safe_headers["Authorization"] = "Bearer ********"
        msg_preview = []
        try:
            for i, m in enumerate(messages[:3]):
                role = m.get("role", "")
                content = (m.get("content", "") or "")
                preview = content[:200].replace("\n", " ")
                msg_preview.append(f"[{i}:{role}] {len(content)} chars :: {preview}...")
        except Exception:
            msg_preview.append("<unprintable messages>")
        logger.debug(
            "OpenRouter request: url=%s timeout=%s model=%s temp=%.3f max_tokens=%s seed=%s reasoning=%s msgs=%d\nheaders=%s\npayload_keys=%s\nmessages_preview=\n%s",
            OPENROUTER_URL,
            timeout,
            model,
            temperature,
            max_tokens,
            seed,
            bool(reasoning_enabled),
            len(messages),
            safe_headers,
            list(payload.keys()),
            "\n".join(msg_preview),
        )

    backoff = 1.5
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            logger.debug("OpenRouter response: status=%s attempt=%s", resp.status_code, attempt)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices in response: {json.dumps(data)[:400]}")
                content = choices[0]["message"]["content"]
                logger.debug("OpenRouter success: content_len=%s preview=%r", len(content or ""), (content or "")[:200])
                return content.strip()
            elif resp.status_code in (429, 529, 500, 503, 504):
                # Rate limit / transient
                sleep_s = backoff ** (attempt + 1) + random.random()
                logger.warning(
                    "Transient HTTP %s; retrying in %.2fs (attempt %d/%d). Body preview: %r",
                    resp.status_code, sleep_s, attempt + 1, max_retries, (resp.text or "")[:300]
                )
                time.sleep(sleep_s)
                continue
            else:
                body_preview = (resp.text or "")[:500]
                logger.error("HTTP %s error body preview: %r", resp.status_code, body_preview)
                raise RuntimeError(f"HTTP {resp.status_code}: {body_preview}")
        except requests.RequestException as e:
            sleep_s = backoff ** (attempt + 1) + random.random()
            logger.warning("RequestException on attempt %d/%d: %s; sleeping %.2fs", attempt + 1, max_retries, e, sleep_s)
            time.sleep(sleep_s)
            if attempt == max_retries - 1:
                raise e
    raise RuntimeError("OpenRouter call failed after retries.")

# ----------------------------
# Helpers
# ----------------------------
def ascii_filter(text: str) -> str:
    return (text or "").encode("ascii", "ignore").decode("ascii")

def normalize_ws(s: str) -> str:
    return "\n".join([line.rstrip() for line in (s or "").strip().splitlines()]).strip()

def load_corpus(csv_path: str) -> pd.DataFrame:
    """
    Tries standard CSV first; falls back to TSV if needed.
    Ensures required columns exist.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        # Fallback to tab-delimited
        df = pd.read_csv(csv_path, sep="\t", engine="python")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}\nColumns found: {list(df.columns)}")
    return df

def build_messages(system_text: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

def transcribe_article(model: str, article_text: str, seed: int, temperature: float) -> str:
    user_prompt = TRANSCRIBE_PROMPT.format(article=article_text)
    msgs = build_messages(SYSTEM_BASE, user_prompt)
    return call_openrouter_chat(
        model=model,
        messages=msgs,
        seed=seed,
        temperature=temperature,
        max_tokens=2048,
        reasoning_enabled=False,
    )

def rewrite_from_transcription(model: str, transcription: str, seed: int, temperature: float) -> str:
    user_prompt = REWRITE_PROMPT.format(transcription=transcription)
    msgs = build_messages(SYSTEM_BASE, user_prompt)
    return call_openrouter_chat(
        model=model,
        messages=msgs,
        seed=seed,
        temperature=temperature,
        max_tokens=2048,
        reasoning_enabled=False,
    )

def judge_pair(
    model: str,
    a_text: str,
    b_text: str,
    criterion_desc: str,  # e.g., "more well-written"
    seed: int,
    temperature: float,
) -> Dict[str, Any]:
    # Build with f-strings to avoid .format() interpreting JSON braces in JUDGE_INSTRUCTIONS
    prompt = (
        f"Based on the two articles below, judge which one is {criterion_desc}.\n\n"
        f"=== ARTICLE A ===\n{a_text}\n\n"
        f"=== ARTICLE B ===\n{b_text}\n\n"
        f"{JUDGE_INSTRUCTIONS}"
    )
    msgs = build_messages(SYSTEM_BASE, prompt)
    raw = call_openrouter_chat(
        model=model,
        messages=msgs,
        seed=seed,
        temperature=temperature,
        max_tokens=512,
        reasoning_enabled=False,
    )
    # Try to parse JSON robustly
    verdict = {"winner": "tie", "confidence": 0.5, "justification": "n/a", "raw": raw}
    try:
        # Extract the first {...} object in case of extra text
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(raw[start:end+1])
            winner = str(j.get("winner", "")).strip().lower()
            if winner in ("a", "b", "tie"):
                verdict["winner"] = winner
            if isinstance(j.get("confidence", None), (int, float)):
                verdict["confidence"] = float(j["confidence"])
            if isinstance(j.get("justification", None), str):
                verdict["justification"] = j["justification"].strip()
    except Exception:
        logging.getLogger("deepseek").warning("Failed to parse JSON verdict; raw reply preview: %r", (raw or "")[:400])
    return verdict

def make_grouped_bar_chart(
    out_png: Path,
    well_ai: int, well_orig: int, well_tie: int,
    hum_ai: int, hum_orig: int, hum_tie: int
):
    labels = ["Well-written", "Human-written"]
    counts_ai = [well_ai, hum_ai]
    counts_orig = [well_orig, hum_orig]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width/2 for i in x], counts_orig, width, label="Original chosen")
    plt.bar([i + width/2 for i in x], counts_ai, width, label="AI chosen")

    plt.xticks(list(x), labels)
    plt.ylabel("Count")
    plt.title(f"Judging Results (ties: well={well_tie}, human={hum_tie})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./fixed.csv", help="Path to corpus CSV/TSV")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model id")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--max-chars", type=int, default=20000, help="Truncate article text to this many characters")
    parser.add_argument("--outdir", type=str, default="deepseek_outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    # Configure root at INFO to avoid noisy library DEBUG logs (e.g., matplotlib findfont)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Our app logger can be elevated to DEBUG independently
    logger = logging.getLogger("deepseek")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # Silence noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "per_article_results.jsonl"
    csv_path = outdir / "per_article_results.csv"
    chart_path = outdir / "results_bar_chart.png"

    logger.info("Starting run: model=%s seed=%s temp=%.3f csv=%s max_rows=%s max_chars=%s outdir=%s",
                args.model, args.seed, args.temperature, args.csv, args.max_rows, args.max_chars, outdir)

    df = load_corpus(args.csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    results: List[Dict[str, Any]] = []

    # Aggregates
    agg = {
        "well_ai": 0, "well_orig": 0, "well_tie": 0,
        "hum_ai": 0, "hum_orig": 0, "hum_tie": 0,
    }

    for idx, row in df.iterrows():
        try:
            article_id = str(row.get("Article ID", idx))
            title = str(row.get("Article Title", "") or "")
            portal = str(row.get("Portal Name", "") or "")
            original_text = str(row.get("Article Text", "") or "").strip()

            if not original_text:
                print(f"[skip] Empty Article Text at index {idx}", file=sys.stderr)
                logging.getLogger("deepseek").info("Skipping index %s due to empty article text", idx)
                continue

            if args.max_chars and len(original_text) > args.max_chars:
                logger.debug("Truncating article_id=%s from %d to %d chars", article_id, len(original_text), args.max_chars)
                original_text = original_text[:args.max_chars]

            logger.info("Index %s article_id=%s portal=%s title_len=%d text_len=%d", idx, article_id, portal, len(title), len(original_text))

            # Step 1: Transcription
            logger.debug("Transcription step starting...")
            transcription = transcribe_article(
                model=args.model,
                article_text=original_text,
                seed=args.seed + idx * 3 + 1,
                temperature=args.temperature,
            )
            logger.debug("Transcription done: %d chars. Preview=%r", len(transcription or ""), (transcription or "")[:200])

            # Step 2: Rewrite
            logger.debug("Rewrite step starting...")
            ai_article = rewrite_from_transcription(
                model=args.model,
                transcription=transcription,
                seed=args.seed + idx * 3 + 2,
                temperature=args.temperature,
            )
            ai_article_ascii = ascii_filter(ai_article)
            original_ascii = ascii_filter(original_text)
            logger.debug("Rewrite done: ai_len=%d orig_len=%d", len(ai_article_ascii or ""), len(original_ascii or ""))

            # Step 3: Randomize order for judging (deterministic per idx)
            rng = random.Random(args.seed + idx)
            order = ["original", "ai"]
            rng.shuffle(order)
            A_text = original_ascii if order[0] == "original" else ai_article_ascii
            B_text = original_ascii if order[1] == "original" else ai_article_ascii

            # Step 4 & 5: Two judgments
            per_article = {
                "article_id": article_id,
                "portal": portal,
                "title": title,
                "order_A": order[0],
                "order_B": order[1],
                "transcription_preview": transcription[:300],
                "ai_article_preview": ai_article_ascii[:300],
            }

            # Criterion 1: Well-written
            logger.debug("Judging criterion 'well-written'...")
            v1 = judge_pair(
                model=args.model,
                a_text=A_text,
                b_text=B_text,
                criterion_desc="more well-written",
                seed=args.seed + idx * 3 + 3,
                temperature=args.temperature,
            )
            per_article["well_written_verdict_raw"] = v1.get("raw", "")
            per_article["well_written_winner_presented"] = v1["winner"]
            per_article["well_written_confidence"] = v1["confidence"]
            per_article["well_written_justification"] = v1["justification"]

            # Map winner back to {original|ai|tie}
            if v1["winner"] == "a":
                winner_kind = order[0]
            elif v1["winner"] == "b":
                winner_kind = order[1]
            else:
                winner_kind = "tie"
            per_article["well_written_winner_kind"] = winner_kind

            if winner_kind == "ai":
                agg["well_ai"] += 1
            elif winner_kind == "original":
                agg["well_orig"] += 1
            else:
                agg["well_tie"] += 1

            # Criterion 2: Human-written
            logger.debug("Judging criterion 'human-written'...")
            v2 = judge_pair(
                model=args.model,
                a_text=A_text,
                b_text=B_text,
                criterion_desc="more human-written",
                seed=args.seed + idx * 3 + 4,
                temperature=args.temperature,
            )
            per_article["human_written_verdict_raw"] = v2.get("raw", "")
            per_article["human_written_winner_presented"] = v2["winner"]
            per_article["human_written_confidence"] = v2["confidence"]
            per_article["human_written_justification"] = v2["justification"]

            if v2["winner"] == "a":
                winner_kind2 = order[0]
            elif v2["winner"] == "b":
                winner_kind2 = order[1]
            else:
                winner_kind2 = "tie"
            per_article["human_written_winner_kind"] = winner_kind2

            if winner_kind2 == "ai":
                agg["hum_ai"] += 1
            elif winner_kind2 == "original":
                agg["hum_orig"] += 1
            else:
                agg["hum_tie"] += 1

            results.append(per_article)
            print(f"[ok] {article_id} :: well={winner_kind} :: human={winner_kind2}")
            logger.info("Completed index %s article_id=%s :: well=%s :: human=%s", idx, article_id, winner_kind, winner_kind2)

            # Write JSONL incrementally (useful for long runs)
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(per_article, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"[error] index={idx} error={e}", file=sys.stderr)
            logging.getLogger("deepseek").exception("Failure processing index %s article_id=%s", idx, str(row.get("Article ID", idx)))
            continue

    # Save CSV summary
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    # Chart
    make_grouped_bar_chart(
        chart_path,
        agg["well_ai"], agg["well_orig"], agg["well_tie"],
        agg["hum_ai"], agg["hum_orig"], agg["hum_tie"],
    )

    # Print quick aggregate summary
    print("\n=== Aggregate Summary ===")
    print(json.dumps(agg, indent=2))
    print(f"\nSaved:\n- {jsonl_path}\n- {csv_path}\n- {chart_path}")

if __name__ == "__main__":
    main()
