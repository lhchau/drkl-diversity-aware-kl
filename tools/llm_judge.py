"""LLM-as-a-judge score of one evaluation run (Table 3, prompt in Appendix G).

The judge rates the model's response and the reference answer separately, with the
MT-Bench single-answer prompt (Zheng et al., 2023), on a 1-10 scale. The score is
100 * sum(response ratings) / sum(reference ratings) over the examples where both got a rating.

    python tools/llm_judge.py data/vicuna results/qwen3-4b/drkl-gamma1.5-seed10/best_rougeL/eval/vicuna/seed10

Needs `pip install anthropic` and an API key in ANTHROPIC_API_KEY. Table 3 used Claude
Haiku 4.5 on 100 random examples of Self-Instruct, Vicuna and UnNI.
"""
import argparse
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic

SYSTEM = ("Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant "
          "to the user question displayed below. Your evaluation should consider factors such as the helpfulness, "
          "relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by "
          "providing a short explanation. Be as objective as possible. After providing your explanation, you must "
          'rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: '
          '"Rating: [[5]]".')
# judges do not always write "[[5]]" literally: also accept "[[rating: 5]]" and "[[rating]]5[[/rating]]"
RATING = re.compile(r"\[\[\s*(?:rating\s*:?\s*)?(\d+(?:\.\d+)?)\s*\]\]|\[\[\s*rating\s*\]\]\s*(\d+(?:\.\d+)?)", re.I)


def rate(client, model, question, answer):
    """The judge's 1-10 rating of `answer`, or None if it refused or gave no parsable rating."""
    message = client.messages.create(
        model=model, max_tokens=1024, system=SYSTEM,
        messages=[{"role": "user", "content": f"[Question]\n{question}\n\n[The Start of Assistant's Answer]\n"
                                              f"{answer}\n[The End of Assistant's Answer]"}])
    if message.stop_reason == "refusal":
        return None
    match = RATING.search("".join(block.text for block in message.content if block.type == "text"))
    return float(match.group(1) or match.group(2)) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="benchmark directory with valid.jsonl, e.g. data/vicuna")
    parser.add_argument("run_dir", help="evaluate.py output directory with answers.jsonl")
    parser.add_argument("--model", default="claude-haiku-4-5", help="judge model (Table 3: Claude Haiku 4.5)")
    parser.add_argument("--num", type=int, default=100, help="random examples to judge, 0 for all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "valid.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    with open(os.path.join(args.run_dir, "answers.jsonl")) as f:
        responses = [json.loads(line)["text"] for line in f]
    assert len(responses) == len(rows), f"{len(responses)} responses for {len(rows)} prompts"
    indices = list(range(len(rows)))
    if 0 < args.num < len(rows):
        indices = sorted(random.Random(args.seed).sample(indices, args.num))

    client = anthropic.Anthropic(max_retries=8)  # the SDK retries rate limits and server errors
    jobs = []
    for i in indices:
        row = rows[i]
        question = f"{row['instruction']}\n\n{row['input']}" if row.get("input") else row["instruction"]
        reference = row["output"][0] if isinstance(row["output"], list) else row["output"]
        jobs += [(question, responses[i]), (question, reference)]
    with ThreadPoolExecutor(args.workers) as pool:
        ratings = list(pool.map(lambda job: rate(client, args.model, *job), jobs))

    records = [{"index": i, "response": r, "reference": g} for i, r, g in zip(indices, ratings[::2], ratings[1::2])]
    rated = [r for r in records if r["response"] is not None and r["reference"] is not None]
    score = 100 * sum(r["response"] for r in rated) / max(sum(r["reference"] for r in rated), 1)
    with open(os.path.join(args.run_dir, "judge.json"), "w") as f:
        json.dump({"model": args.model, "score": round(score, 2), "rated": len(rated), "ratings": records}, f, indent=2)
    print(f"{args.run_dir}: judge score {score:.2f} ({len(rated)}/{len(records)} examples rated by {args.model})")


if __name__ == "__main__":
    main()
