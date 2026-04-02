"""
LLM-as-judge coverage scorer.

For each event, sends all claims + relevant paragraphs to Claude and gets back
a coverage matrix: which paragraphs cover which claims.

Usage:
    python scripts/llm_scorer.py                    # run all 147 events
    python scripts/llm_scorer.py --events 46 75     # run specific events
    python scripts/llm_scorer.py --resume            # resume from checkpoint

Output: data/processed/llm_coverage_labels.json
"""

import json
import os
import time
import argparse
import anthropic
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "processed"

# Load API key from .env file
ENV_PATH = PROJECT_ROOT.parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                os.environ[key.strip()] = val
    # Map ANTHROPIC_API to ANTHROPIC_API_KEY if needed
    if "ANTHROPIC_API" in os.environ and "ANTHROPIC_API_KEY" not in os.environ:
        os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API"]

SYSTEM_PROMPT = """You are an expert annotator for a news retrieval research project.
Your job: given a set of paragraphs from news articles about an event, and a set of perspective claims,
determine which paragraphs COVER which claims.

A paragraph COVERS a claim if a reader of that paragraph would come away understanding this perspective.
- Being about the same TOPIC is NOT enough. The paragraph must convey the same POINT.
- "Covers" means the perspective is expressed or directly supported, not just vaguely related.
- If a claim is a specific quote from an expert, the paragraph must contain that quote or its core argument.
- When genuinely unsure, lean toward false (conservative)."""

USER_PROMPT_TEMPLATE = """Event: {headline}

=== CLAIMS ===
{claims_text}

=== PARAGRAPHS ===
{paragraphs_text}

For each (claim, paragraph) pair, determine if the paragraph covers the claim.

Respond with ONLY a JSON object mapping claim_id to a list of paragraph_ids that cover it.
If a claim is not covered by any paragraph, map it to an empty list.
Example format:
{{"claim_1": ["para_a", "para_b"], "claim_2": [], "claim_3": ["para_c"]}}

Be conservative: only include a paragraph if it clearly expresses the claim's perspective, not just the same topic."""


def build_prompt(event):
    """Build the prompt for one event."""
    claims_text = ""
    for c in event["claims"]:
        claims_text += f'\n[{c["claim_id"]}] {c["text"]}'

    rel_paragraphs = [p for p in event["paragraphs"] if p["relevant"] == 1]
    paragraphs_text = ""
    for p in rel_paragraphs:
        paragraphs_text += f'\n[{p["paragraph_id"]}]\n{p["text"]}\n'

    return USER_PROMPT_TEMPLATE.format(
        headline=event["headline"],
        claims_text=claims_text,
        paragraphs_text=paragraphs_text,
    )


def score_event(client, event, model="claude-sonnet-4-5-20250929", max_retries=3):
    """Score one event, return coverage dict."""
    prompt = build_prompt(event)
    claim_ids = [c["claim_id"] for c in event["claims"]]

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            text = response.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            coverage = json.loads(text)

            # Validate: ensure all claim_ids are present
            for cid in claim_ids:
                if cid not in coverage:
                    coverage[cid] = []

            # Ensure paragraph IDs are valid
            valid_pids = {p["paragraph_id"] for p in event["paragraphs"]}
            for cid in coverage:
                coverage[cid] = [pid for pid in coverage[cid] if pid in valid_pids]

            return coverage, response.usage

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  Attempt {attempt+1} failed (parse error): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            continue
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        except anthropic.APIError as e:
            print(f"  API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", nargs="*", help="Specific event IDs to run")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Model to use",
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for majority vote")
    args = parser.parse_args()

    # Load data
    with open(DATA / "coverage_data.json") as f:
        all_events = json.load(f)

    # Filter events if specified
    if args.events:
        all_events = [e for e in all_events if e["dsglobal_id"] in args.events]
        print(f"Running {len(all_events)} specified events")
    else:
        print(f"Running all {len(all_events)} events")

    # Load checkpoint if resuming
    checkpoint_path = DATA / "llm_coverage_checkpoint.json"
    if args.resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        done_ids = set(results.keys())
        print(f"Resuming: {len(done_ids)} events already done")
    else:
        results = {}
        done_ids = set()

    client = anthropic.Anthropic()

    total_input = 0
    total_output = 0
    total_events = len(all_events)

    for i, event in enumerate(all_events):
        eid = event["dsglobal_id"]
        if eid in done_ids:
            continue

        n_claims = event["n_claims"]
        n_rel = event["n_relevant"]
        print(
            f"[{i+1}/{total_events}] Event {eid}: {n_claims} claims x {n_rel} rel paragraphs = {n_claims * n_rel} pairs"
        )

        if args.runs == 1:
            coverage, usage = score_event(client, event, model=args.model)
            if coverage is None:
                print(f"  FAILED — skipping")
                continue
            results[eid] = coverage
            if usage:
                total_input += usage.input_tokens
                total_output += usage.output_tokens
        else:
            # Multiple runs for majority vote
            all_runs = []
            for run in range(args.runs):
                coverage, usage = score_event(client, event, model=args.model)
                if coverage:
                    all_runs.append(coverage)
                    if usage:
                        total_input += usage.input_tokens
                        total_output += usage.output_tokens

            if len(all_runs) < 2:
                print(f"  FAILED — fewer than 2 successful runs")
                continue

            # Majority vote
            merged = {}
            for cid in event["claims"]:
                cid = cid["claim_id"]
                # Count how many runs included each paragraph for this claim
                pid_counts = {}
                for run_coverage in all_runs:
                    for pid in run_coverage.get(cid, []):
                        pid_counts[pid] = pid_counts.get(pid, 0) + 1
                # Majority: included in >50% of runs
                threshold = len(all_runs) / 2
                merged[cid] = [
                    pid for pid, count in pid_counts.items() if count > threshold
                ]
            results[eid] = merged

        # Print summary
        n_covered = sum(1 for cid, pids in results[eid].items() if len(pids) > 0)
        print(
            f"  → {n_covered}/{n_claims} claims covered ({n_covered/n_claims*100:.0f}%)"
        )

        # Checkpoint every 10 events
        if (i + 1) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)
            print(f"  [checkpoint saved: {len(results)} events]")
            cost_estimate = (total_input * 3 + total_output * 15) / 1_000_000
            print(
                f"  [tokens so far: {total_input:,} in / {total_output:,} out ≈ ${cost_estimate:.2f}]"
            )

        # Small delay to avoid rate limits
        time.sleep(0.5)

    # Save final results
    output_path = DATA / "llm_coverage_labels.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save in a flat format for easy analysis
    flat = []
    for eid, coverage in results.items():
        for cid, pids in coverage.items():
            for ev in all_events:
                if ev["dsglobal_id"] == eid:
                    all_rel_pids = [
                        p["paragraph_id"]
                        for p in ev["paragraphs"]
                        if p["relevant"] == 1
                    ]
                    for pid in all_rel_pids:
                        flat.append(
                            {
                                "event_id": eid,
                                "claim_id": cid,
                                "paragraph_id": pid,
                                "covered": pid in pids,
                            }
                        )
                    break

    with open(DATA / "llm_coverage_flat.json", "w") as f:
        json.dump(flat, f)

    # Summary
    print(f"\n=== DONE ===")
    print(f"Events scored: {len(results)}")
    total_claims_scored = sum(len(v) for v in results.values())
    total_covered = sum(
        sum(1 for pids in v.values() if pids) for v in results.values()
    )
    print(f"Total claims: {total_claims_scored}")
    print(f"Claims with ≥1 covering paragraph: {total_covered}")
    if total_claims_scored > 0:
        print(
            f"Overall coverage rate: {total_covered/total_claims_scored*100:.1f}%"
        )
    else:
        print("No events scored successfully.")
    cost_estimate = (total_input * 3 + total_output * 15) / 1_000_000
    print(
        f"Total tokens: {total_input:,} in / {total_output:,} out ≈ ${cost_estimate:.2f}"
    )
    print(f"Saved to: {output_path}")
    print(f"Flat format: {DATA / 'llm_coverage_flat.json'}")


if __name__ == "__main__":
    main()
