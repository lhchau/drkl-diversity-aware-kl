"""Fidelity (ROUGE-L, exact match) and diversity (Distinct-2, Self-BLEU) of generated responses."""
import string

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_punctuation = set(string.punctuation)


def normalize_answer(s):
    """Lowercase, drop punctuation and collapse whitespace."""
    s = "".join(ch for ch in s.lower() if ch not in _punctuation)
    return " ".join(s.split())


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge(prediction, ground_truth):
    return _scorer.score(prediction=prediction, target=ground_truth)["rougeL"].fmeasure


def compute_metrics(predictions, references):
    """predictions: list[str]; references: list[list[str]]. Scores are in [0, 100]."""
    n = min(len(predictions), len(references))
    em = sum(max(exact_match(p, g) for g in gold) for p, gold in zip(predictions[:n], references[:n]))
    rouge_l = sum(max(rouge(p, g) for g in gold) for p, gold in zip(predictions[:n], references[:n]))
    return {"exact_match": round(100.0 * em / n, 4), "rougeL": round(100.0 * rouge_l / n, 4)}


def distinct_2(texts):
    """Unique / total word bigrams over a group of responses (Li et al., 2016)."""
    bigrams = [b for t in texts for b in zip(t.split()[:-1], t.split()[1:])]
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def neg_self_bleu(texts):
    """Minus the mean BLEU of each response against the other ones (Zhu et al., 2018); higher is more diverse."""
    tokens = [t.split() for t in texts]
    if len(tokens) < 2:
        return 0.0
    smoothing = SmoothingFunction().method1
    scores = [sentence_bleu(tokens[:i] + tokens[i + 1:], hyp, smoothing_function=smoothing)
              for i, hyp in enumerate(tokens) if hyp]
    return -float(np.mean(scores)) if scores else 0.0


def diversity_metrics(texts, group_size):
    """Distinct-2 and negative Self-BLEU, each averaged over consecutive groups of responses."""
    groups = [texts[i:i + group_size] for i in range(0, len(texts), group_size)]
    return {"distinct2": round(float(np.mean([distinct_2(g) for g in groups])), 4),
            "neg_self_bleu": round(float(np.mean([neg_self_bleu(g) for g in groups])), 4)}
