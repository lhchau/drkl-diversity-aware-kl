"""Print mean +- std over seeds of every metric, per benchmark, under an eval directory.

    python tools/summarize_eval.py results/qwen3-4b/drkl-seed10/best_rougeL/eval
"""
import glob
import json
import os
import statistics
import sys

BENCHMARKS = ["dolly", "self-inst", "vicuna", "sinst/11_", "uinst/11_"]  # column order of Table 1
METRICS = ["rougeL", "distinct2", "neg_self_bleu", "entropy", "confidence"]


def summarize(eval_dir):
    scores = {}
    for path in glob.glob(os.path.join(eval_dir, "**", "seed*", "metrics.json"), recursive=True):
        bench = os.path.relpath(os.path.dirname(os.path.dirname(path)), eval_dir)
        with open(path) as f:
            for k, v in json.load(f).items():
                scores.setdefault(bench, {}).setdefault(k, []).append(v)

    print(f"{'benchmark':12s}" + "".join(f"{k:>18s}" for k in METRICS) + "   seeds")
    means = {k: [] for k in METRICS}
    for bench in sorted(scores, key=lambda b: BENCHMARKS.index(b) if b in BENCHMARKS else len(BENCHMARKS)):
        row = f"{bench:12s}"
        for k in METRICS:
            values = scores[bench].get(k)
            if not values:
                row += f"{'-':>18s}"
                continue
            means[k].append(statistics.mean(values))
            d = 2 if k == "rougeL" else 3
            row += f"{f'{means[k][-1]:.{d}f} +- {statistics.pstdev(values):.{d}f}':>18s}"
        print(f"{row}   {len(scores[bench]['rougeL'])}")
    print(f"{'average':12s}" + "".join(f"{statistics.mean(v):>18.{2 if k == 'rougeL' else 3}f}" if v else f"{'-':>18s}"
                                       for k, v in means.items()))
    return scores


if __name__ == "__main__":
    summarize(sys.argv[1])
