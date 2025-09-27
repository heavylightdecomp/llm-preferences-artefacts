#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment: Do LLMs associate "good" opinion writing with human-style writing?

This version implements:
  - Generation-only model: DeepSeek V3.1 (transcribe + rewrite).
  - Judge models: Grok 4, Claude 4 Sonnet, GPT-5 (each runs full judging).
  - Hard cap: 4096 output tokens per API call (generation + judging).
  - Per-model CSV/JSONL outputs + per-model matplotlib bar charts.
  - A detailed, multi-model summary printed to console and exported to outputs/summary.txt.

Dependencies:
  pip install pandas matplotlib requests python-dotenv

Environment:
  export OPENROUTER_API_KEY=...    # or place in .env

Usage:
  python experiment_openrouter_opinion_multi_judges.py \
      --csv ./fixed.csv \
      --seed 42 \
      --max-rows 100 \
      --verbose

Notes:
  - Generation model: deepseek/deepseek-chat-v3.1
  - Default judge IDs are provider-sensitive; this script tries aliases if the first fails.
  - We fix the A/B order per article (seeded), and reuse that same order for every judge model.
"""

import os
import sys
import csv
import json
import time
import random
import argparse
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

GEN_MODEL = "deepseek/deepseek-chat-v3.1"  # generation-only (transcribe + rewrite)
TOKEN_LIMIT = 4096
DEFAULT_SEED = 42

# Judge models (friendly key -> preferred id with fallback aliases in order)
JUDGE_MODEL_ALIASES: Dict[str, List[str]] = {
    "grok4": ["x-ai/grok-4"],
    "claude4sonnet": ["anthropic/claude-sonnet-4"],
    "gpt5": ["openai/gpt-5"],
}

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
    "You are a careful rater. Read the two articles. Use minimal internal reasoning (low effort) and do not over-deliberate. Respond with strict JSON:\n\n"
    '{\n  "winner": "A" or "B" or "tie",\n  "confidence": float between 0 and 1,\n'
    '  "justification": "one short sentence without revealing chain-of-thought"\n}\n\n'
    "Do not output anything except the JSON object."
)

EXPECTED_COLUMNS = [
    "Article ID", "Portal Name", "Last Update", "Source URL",
    "Author Names", "Article Title", "Article Text", "ArticleSummary"
]

# ----------------------------
# HTTP / OpenRouter
# ----------------------------
def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    h = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.local/experiments",
        "X-Title": "Opinion-LLM-Judging-Experiment",
    }
    if extra:
        h.update(extra)
    return h

def call_openrouter_chat(
    model: str,
    messages: List[Dict[str, str]],
    seed: Optional[int],
    temperature: float,
    max_tokens: int = TOKEN_LIMIT,
    reasoning_enabled: bool = False,
    max_retries: int = 6,
    timeout: int = 120,
) -> str:
    """Call OpenRouter Chat Completions endpoint and return assistant content."""
    logger = logging.getLogger("exp")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "seed": seed,
        "top_p": 0.95,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        # For models that require reasoning (e.g., Grok 4), enable it and request low effort
        "reasoning": ({"enabled": True, "effort": "low"} if reasoning_enabled else {"enabled": False}),
    }

    backoff = 1.5
    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=_headers(), json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"No choices in response: {json.dumps(data)[:400]}")
                content = choices[0]["message"]["content"]
                return (content or "").strip()
            elif resp.status_code in (429, 529, 500, 503, 504):
                sleep_s = backoff ** (attempt + 1) + random.random()
                logger.warning("Transient HTTP %s (attempt %d/%d). Sleeping %.2fs",
                               resp.status_code, attempt + 1, max_retries, sleep_s)
                time.sleep(sleep_s)
                continue
            else:
                body_preview = (resp.text or "")[:500]
                raise RuntimeError(f"HTTP {resp.status_code}: {body_preview}")
        except requests.RequestException as e:
            sleep_s = backoff ** (attempt + 1) + random.random()
            logger.warning("RequestException (attempt %d/%d): %s; sleeping %.2fs",
                           attempt + 1, max_retries, e, sleep_s)
            time.sleep(sleep_s)
            if attempt == max_retries - 1:
                raise e
    raise RuntimeError("OpenRouter call failed after retries.")

# ----------------------------
# Helpers
# ----------------------------
def ascii_filter(text: str) -> str:
    return (text or "").encode("ascii", "ignore").decode("ascii")

def load_corpus(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep="\t", engine="python")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}\nFound: {list(df.columns)}")
    return df

def build_messages(system_text: str, user_text: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": system_text},
            {"role": "user", "content": user_text}]

def transcribe_article(article_text: str, seed: int, temperature: float) -> str:
    user_prompt = TRANSCRIBE_PROMPT.format(article=article_text)
    msgs = build_messages(SYSTEM_BASE, user_prompt)
    return call_openrouter_chat(
        model=GEN_MODEL,
        messages=msgs,
        seed=seed,
        temperature=temperature,
        max_tokens=TOKEN_LIMIT,
        reasoning_enabled=False,
    )

def rewrite_from_transcription(transcription: str, seed: int, temperature: float) -> str:
    user_prompt = REWRITE_PROMPT.format(transcription=transcription)
    msgs = build_messages(SYSTEM_BASE, user_prompt)
    return call_openrouter_chat(
        model=GEN_MODEL,
        messages=msgs,
        seed=seed,
        temperature=temperature,
        max_tokens=TOKEN_LIMIT,
        reasoning_enabled=False,
    )

def judge_pair(
    judge_model: str,
    a_text: str,
    b_text: str,
    criterion_desc: str,  # "more well-written" | "more human-written"
    seed: int,
    temperature: float,
) -> Dict[str, Any]:
    prompt = (
        f"Based on the two articles below, judge which one is {criterion_desc}.\n\n"
        f"=== ARTICLE A ===\n{a_text}\n\n"
        f"=== ARTICLE B ===\n{b_text}\n\n"
        f"{JUDGE_INSTRUCTIONS}"
    )
    msgs = build_messages(SYSTEM_BASE, prompt)
    raw = call_openrouter_chat(
        model=judge_model,
        messages=msgs,
        seed=seed,
        temperature=temperature,
        max_tokens=TOKEN_LIMIT,
        reasoning_enabled=True,
    )
    verdict = {"winner": "tie", "confidence": 0.5, "justification": "n/a", "raw": raw}
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(raw[start:end+1])
            w = str(j.get("winner", "")).strip().lower()
            if w in ("a", "b", "tie"):
                verdict["winner"] = w
            if isinstance(j.get("confidence", None), (int, float)):
                verdict["confidence"] = float(j["confidence"])
            if isinstance(j.get("justification", None), str):
                verdict["justification"] = j["justification"].strip()
    except Exception:
        logging.getLogger("exp").warning("Failed to parse JSON verdict; raw preview: %r", (raw or "")[:400])
    return verdict

def make_bar_chart(
    out_png: Path,
    model_label: str,
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
    plt.bar([i + width/2 for i in x], counts_ai,   width, label="AI chosen")
    plt.xticks(list(x), labels)
    plt.ylabel("Count")
    plt.title(f"{model_label} — Judging Results (ties: well={well_tie}, human={hum_tie})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./fixed.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--verbose", action="store_true")
    # Optional custom selection of judge models (comma-separated keys that exist in JUDGE_MODEL_ALIASES)
    parser.add_argument("--judge", type=str, default="grok4,claude4sonnet,gpt5",
                        help="Comma-separated list of judge keys: grok4,claude4sonnet,gpt5")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("exp")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    for noisy in ["matplotlib", "PIL", "fontTools"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve which judges to run
    judge_keys = [k.strip().lower() for k in args.judge.split(",") if k.strip()]
    for k in judge_keys:
        if k not in JUDGE_MODEL_ALIASES:
            raise ValueError(f"Unknown judge key '{k}'. Allowed: {list(JUDGE_MODEL_ALIASES)}")

    # Validate/resolve judge ids via a cheap probe
    resolved_judges: Dict[str, str] = {}
    for jk in judge_keys:
        # Use the first alias directly without probing
        resolved_judges[jk] = JUDGE_MODEL_ALIASES[jk][0]

    # Load corpus
    df = load_corpus(args.csv)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    # ---------- Phase 1: Generate AI articles with DeepSeek ----------
    # We precompute once and reuse across every judge.
    generation_records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        try:
            article_id = str(row.get("Article ID", idx))
            title = str(row.get("Article Title", "") or "")
            portal = str(row.get("Portal Name", "") or "")
            original_text = str(row.get("Article Text", "") or "").strip()

            if not original_text:
                logger.info("[skip] Empty Article Text at index %s", idx)
                continue

            if args.max_chars and len(original_text) > args.max_chars:
                original_text = original_text[:args.max_chars]

            logger.info("Generate idx=%s id=%s portal=%s len(title)=%d len(text)=%d",
                        idx, article_id, portal, len(title), len(original_text))

            # Step 1: Transcribe
            transcription = transcribe_article(
                article_text=original_text,
                seed=args.seed + idx * 5 + 1,
                temperature=args.temperature,
            )

            # Step 2: Rewrite
            ai_article = rewrite_from_transcription(
                transcription=transcription,
                seed=args.seed + idx * 5 + 2,
                temperature=args.temperature,
            )

            # Step 3: ASCII filter both
            ai_article_ascii = ascii_filter(ai_article)
            original_ascii = ascii_filter(original_text)

            # Step 4: Deterministic order per article (shared by all judges)
            rng = random.Random(args.seed + idx)
            order = ["original", "ai"]
            rng.shuffle(order)
            A_text = original_ascii if order[0] == "original" else ai_article_ascii
            B_text = original_ascii if order[1] == "original" else ai_article_ascii

            generation_records.append({
                "idx": idx,
                "article_id": article_id,
                "portal": portal,
                "title": title,
                "order_A": order[0],
                "order_B": order[1],
                "A_text": A_text,
                "B_text": B_text,
                "transcription_preview": (transcription or "")[:500],
                "ai_article_preview": (ai_article_ascii or "")[:500],
            })
        except Exception as e:
            logger.exception("[generation error] idx=%s id=%s error=%s", idx, row.get("Article ID", idx), e)

    if not generation_records:
        print("No articles generated; nothing to judge.")
        return

    # ---------- Phase 2: Per-judge evaluation ----------
    # Aggregates per judge
    all_model_agg: Dict[str, Dict[str, Any]] = {}
    summary_lines: List[str] = []
    summary_lines.append("=== Multi-Model Judging Summary ===")
    summary_lines.append(f"Generation model: {GEN_MODEL}")
    summary_lines.append(f"Judges: {', '.join([f'{k} -> {resolved_judges[k]}' for k in judge_keys])}")
    summary_lines.append(f"Articles evaluated: {len(generation_records)}")
    summary_lines.append(f"Max output tokens per call: {TOKEN_LIMIT}")
    summary_lines.append("")

    for jk in judge_keys:
        judge_id = resolved_judges[jk]
        logger.info("=== Running judge: %s (%s) ===", jk, judge_id)

        # Per-judge output dir and files
        judge_dir = outdir / jk
        judge_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = judge_dir / f"per_article_results_{jk}.jsonl"
        csv_path   = judge_dir / f"per_article_results_{jk}.csv"
        chart_path = judge_dir / f"results_bar_chart_{jk}.png"

        results: List[Dict[str, Any]] = []
        agg = {
            "well_ai": 0, "well_orig": 0, "well_tie": 0,
            "hum_ai": 0,  "hum_orig": 0,  "hum_tie": 0,
            # cross-tab: (well_winner, human_winner)
            "both_ai": 0, "both_orig": 0, "well_ai_human_orig": 0, "well_orig_human_ai": 0, "any_tie": 0,
            "mean_conf_well": 0.0, "mean_conf_human": 0.0, "n_conf_well": 0, "n_conf_human": 0,
        }

        for rec in generation_records:
            try:
                A_text = rec["A_text"]
                B_text = rec["B_text"]

                # Criterion 1: more well-written
                v_well = judge_pair(
                    judge_model=judge_id,
                    a_text=A_text, b_text=B_text,
                    criterion_desc="more well-written",
                    seed=args.seed + rec["idx"] * 5 + 3,
                    temperature=args.temperature,
                )

                # Winner kind mapping back to {original|ai|tie}
                if v_well["winner"] == "a":
                    well_kind = rec["order_A"]
                elif v_well["winner"] == "b":
                    well_kind = rec["order_B"]
                else:
                    well_kind = "tie"

                if well_kind == "ai": agg["well_ai"] += 1
                elif well_kind == "original": agg["well_orig"] += 1
                else: agg["well_tie"] += 1

                if v_well["winner"] in ("a", "b"):
                    agg["mean_conf_well"] += float(v_well["confidence"])
                    agg["n_conf_well"] += 1

                # Criterion 2: more human-written
                v_hum = judge_pair(
                    judge_model=judge_id,
                    a_text=A_text, b_text=B_text,
                    criterion_desc="more human-written",
                    seed=args.seed + rec["idx"] * 5 + 4,
                    temperature=args.temperature,
                )

                if v_hum["winner"] == "a":
                    hum_kind = rec["order_A"]
                elif v_hum["winner"] == "b":
                    hum_kind = rec["order_B"]
                else:
                    hum_kind = "tie"

                if hum_kind == "ai": agg["hum_ai"] += 1
                elif hum_kind == "original": agg["hum_orig"] += 1
                else: agg["hum_tie"] += 1

                if v_hum["winner"] in ("a", "b"):
                    agg["mean_conf_human"] += float(v_hum["confidence"])
                    agg["n_conf_human"] += 1

                # Cross-tabulation
                if well_kind == "ai" and hum_kind == "ai":
                    agg["both_ai"] += 1
                elif well_kind == "original" and hum_kind == "original":
                    agg["both_orig"] += 1
                elif well_kind == "ai" and hum_kind == "original":
                    agg["well_ai_human_orig"] += 1
                elif well_kind == "original" and hum_kind == "ai":
                    agg["well_orig_human_ai"] += 1
                else:
                    # any tie in either dimension
                    agg["any_tie"] += 1

                per_article = {
                    **{k: rec[k] for k in ["article_id", "portal", "title", "order_A", "order_B",
                                           "transcription_preview", "ai_article_preview"]},
                    "well_winner_presented": v_well["winner"],
                    "well_confidence": v_well["confidence"],
                    "well_justification": v_well["justification"],
                    "well_winner_kind": well_kind,

                    "human_winner_presented": v_hum["winner"],
                    "human_confidence": v_hum["confidence"],
                    "human_justification": v_hum["justification"],
                    "human_winner_kind": hum_kind,
                }
                results.append(per_article)

                # Incremental JSONL write
                with jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(per_article, ensure_ascii=False) + "\n")

            except Exception as e:
                logger.exception("[judge error] judge=%s idx=%s id=%s", jk, rec["idx"], rec["article_id"])

        # Finalize per-judge CSV
        if results:
            pd.DataFrame(results).to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

        # Compute means
        if agg["n_conf_well"] > 0:
            agg["mean_conf_well"] = agg["mean_conf_well"] / agg["n_conf_well"]
        if agg["n_conf_human"] > 0:
            agg["mean_conf_human"] = agg["mean_conf_human"] / agg["n_conf_human"]

        # Bar chart
        make_bar_chart(
            chart_path,
            model_label=f"{jk} ({judge_id})",
            well_ai=agg["well_ai"], well_orig=agg["well_orig"], well_tie=agg["well_tie"],
            hum_ai=agg["hum_ai"],   hum_orig=agg["hum_orig"],   hum_tie=agg["hum_tie"],
        )

        all_model_agg[jk] = {
            "judge_id": judge_id,
            "counts": agg,
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "chart": str(chart_path),
        }

        # Pretty per-judge summary
        N = len(generation_records)
        def pct(x): return f"{(100.0*x/N):.1f}%"
        summary_lines.append(f"--- {jk} ({judge_id}) ---")
        summary_lines.append(f"Articles: {N}")
        summary_lines.append(f"Well-written: AI={agg['well_ai']} ({pct(agg['well_ai'])}), "
                             f"Original={agg['well_orig']} ({pct(agg['well_orig'])}), "
                             f"Tie={agg['well_tie']} ({pct(agg['well_tie'])})")
        summary_lines.append(f"Human-written: AI={agg['hum_ai']} ({pct(agg['hum_ai'])}), "
                             f"Original={agg['hum_orig']} ({pct(agg['hum_orig'])}), "
                             f"Tie={agg['hum_tie']} ({pct(agg['hum_tie'])})")
        summary_lines.append(f"Cross: both_ai={agg['both_ai']}, both_orig={agg['both_orig']}, "
                             f"well_ai_human_orig={agg['well_ai_human_orig']}, "
                             f"well_orig_human_ai={agg['well_orig_human_ai']}, any_tie={agg['any_tie']}")
        summary_lines.append(f"Mean confidence: well={agg['mean_conf_well']:.3f} "
                             f"(n={agg['n_conf_well']}), human={agg['mean_conf_human']:.3f} "
                             f"(n={agg['n_conf_human']})")
        summary_lines.append(f"Files: {jsonl_path}, {csv_path}, {chart_path}")
        summary_lines.append("")

    # ---------- Global summary output ----------
    # Save summary.txt
    summary_path = outdir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # Print to console
    print("\n".join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")
    print("Per-judge artifacts:")
    for jk, meta in all_model_agg.items():
        print(f"- {jk}: csv={meta['csv']} jsonl={meta['jsonl']} chart={meta['chart']}")

if __name__ == "__main__":
    main()
