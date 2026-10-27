"""Token-level distillation objectives compared in the paper.

Every loss takes student logits and teacher logits of shape [B, L, V] and the
labels [B, L] (-100 marks positions to ignore), and averages over the valid
positions. Baselines follow DistiLLM (Ko et al., 2024) and the original papers.

Notation of the paper: q is the student, p the teacher and m the target token
(the label). q~_m = (q_m, 1 - q_m) is the binary target-vs-rest distribution and
q^ the distribution renormalized over the non-target tokens (same for p).
"""
import math

import torch
import torch.nn.functional as F

LOSSES = ["sft", "fkl", "rkl", "symkl", "dkd", "js", "sfkl", "srkl", "akl", "ab", "drkl", "srkl_drkl"]


def _valid(logits, teacher_logits, labels):
    """Flatten to [N, V], keeping only the N positions that have a label."""
    valid = labels.reshape(-1) != -100
    return logits.reshape(-1, logits.size(-1))[valid], teacher_logits.reshape(-1, teacher_logits.size(-1))[valid]


def _kl(log_a, log_b):
    """sum_k a_k (log a_k - log b_k) per position; tokens with -inf logits contribute 0."""
    x = log_a.exp() * (log_a - log_b)
    return torch.masked_fill(x, torch.isinf(log_a) | torch.isinf(log_b), 0).sum(-1)


def _binary_kl(a, b):
    """KL((a, 1 - a) || (b, 1 - b)) per position, for a and b of shape [N, 1]."""
    a, b = torch.cat([a, 1 - a], 1), torch.cat([b, 1 - b], 1)
    return (a * (a.clamp(min=1e-12).log() - b.clamp(min=1e-12).log())).sum(1)


def target_split(logits, teacher_logits, labels):
    """q_m, p_m [N, 1] and the non-target log-distributions log q^, log p^ [N, V] (Proposition 1)."""
    target = labels.reshape(-1)
    target = target[target != -100].unsqueeze(1)
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    q_m = F.softmax(logits, dim=-1, dtype=torch.float32).gather(1, target)
    p_m = F.softmax(teacher_logits, dim=-1, dtype=torch.float32).gather(1, target)
    # a -1000 target logit renormalizes the softmax over the non-target tokens
    log_q_hat = F.log_softmax(logits.scatter(1, target, -1000.0), dim=-1, dtype=torch.float32)
    log_p_hat = F.log_softmax(teacher_logits.scatter(1, target, -1000.0), dim=-1, dtype=torch.float32)
    return q_m, p_m, log_q_hat, log_p_hat


def forward_kl(logits, teacher_logits, labels):
    """FKL: KL(p || q), written as cross-entropy (the teacher entropy is a constant)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    return -torch.masked_fill(teacher_probs * student_logprobs, torch.isinf(logits), 0).sum(-1).mean()


def reverse_kl(logits, teacher_logits, labels):
    """RKL: KL(q || p) (Gu et al., 2024)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    x = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    x -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    return -x.sum(-1).mean()


def symmetric_kl(logits, teacher_logits, labels, lam=0.9):
    """Sym-KL: (1 - lam) * FKL + lam * RKL, with DistiLLM's lam = 0.9."""
    return (1 - lam) * forward_kl(logits, teacher_logits, labels) + lam * reverse_kl(logits, teacher_logits, labels)


def decoupled_kd(logits, teacher_logits, labels, alpha=1.0, beta=2.0):
    """DKD (Zhao et al., 2022): alpha * binary target-class FKL + beta * non-target FKL, as cross-entropies."""
    q_m, p_m, log_q_hat, log_p_hat = target_split(logits, teacher_logits, labels)
    q2, p2 = torch.cat([q_m, 1 - q_m], 1), torch.cat([p_m, 1 - p_m], 1)
    target_ce = -(p2 * q2.clamp(min=1e-12).log()).sum(1)
    non_target_ce = -torch.masked_fill(log_p_hat.exp() * log_q_hat, torch.isinf(log_q_hat), 0).sum(1)
    return (alpha * target_ce + beta * non_target_ce).mean()


def js_divergence(logits, teacher_logits, labels, lam=0.9):
    """Generalized JSD with mixture m = (1 - lam) * p + lam * q (Agarwal et al., 2024)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    x = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    x -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    loss = lam * -x.sum(-1).mean()
    x = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    return loss + (1 - lam) * -x.sum(-1).mean()


def skewed_forward_kl(logits, teacher_logits, labels, lam=0.1):
    """SFKL: KL(p || lam * p + (1 - lam) * q) (Ko et al., 2024), as cross-entropy."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(lam * teacher_probs + (1 - lam) * student_probs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    return -torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0).sum(-1).mean()


def skewed_reverse_kl(logits, teacher_logits, labels, lam=0.1):
    """SRKL: KL(q || (1 - lam) * p + lam * q) (Ko et al., 2024)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log((1 - lam) * teacher_probs + lam * student_probs)
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    x = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    x -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    return -x.sum(-1).mean()


def _akl_ratio(teacher_logits, logits, mu=0.5):
    """Head/tail error split of AKL: the head is the smallest set of top teacher tokens holding mass mu."""
    teacher_probs = F.softmax(torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0), dim=-1, dtype=torch.float32)
    student_probs = F.softmax(torch.masked_fill(logits, torch.isinf(logits), 0), dim=-1, dtype=torch.float32).detach()
    sorted_teacher, idx = teacher_probs.sort(dim=-1, descending=True)
    errors = torch.abs(sorted_teacher - student_probs.gather(dim=-1, index=idx))
    tail = torch.cumsum(sorted_teacher, dim=-1) > mu
    tail[:, 0] = False  # the top token always belongs to the head
    head_err = torch.masked_fill(errors, tail, 0.0).sum(-1)
    tail_err = torch.masked_fill(errors, ~tail, 0.0).sum(-1)
    return head_err / (head_err + tail_err), tail_err / (head_err + tail_err)


def _weighted_kl(target_logits, logits, inf_mask, weight):
    """sum_k p_k (log p_k - log q_k) with p from target_logits, weighted per position."""
    target_probs = F.softmax(target_logits, dim=-1, dtype=torch.float32)
    target_logprobs = F.log_softmax(target_logits, dim=-1, dtype=torch.float32)
    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_x = torch.masked_fill(target_probs * target_logprobs, inf_mask, 0).sum(-1)
    x = torch.masked_fill(target_probs * logprobs, inf_mask, 0).sum(-1)
    return ((target_x - x) * weight).mean()


def adaptive_kl(logits, teacher_logits, labels):
    """AKL: head-weighted FKL + tail-weighted RKL (Wu et al., 2025)."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    inf_mask = torch.isinf(logits)
    head, tail = _akl_ratio(teacher_logits, logits)
    return _weighted_kl(teacher_logits, logits, inf_mask, head) + _weighted_kl(logits, teacher_logits, inf_mask, tail)


def ab_divergence(logits, teacher_logits, labels, alpha=0.2, beta=0.7):
    """Alpha-beta divergence (Wang et al., 2025), general case alpha, beta, alpha + beta != 0."""
    logits, teacher_logits = _valid(logits, teacher_logits, labels)
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    apb = alpha + beta
    term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
    term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
    term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
    divergence = -(term1 - term2 - term3) / (alpha * beta)
    return torch.where(torch.isfinite(divergence), divergence, 0.0).mean()


def diversity_aware_reverse_kl(logits, teacher_logits, labels, gamma=1.0, skew=0.0):
    """DRKL (Eq. 9): KL(q~_m || p~_m) + gamma * KL(q^ || p^).

    RKL is the same sum with weight (1 - q_m) instead of gamma (Proposition 1). That weight
    lets the non-target term push the target logit up (Proposition 2); a constant removes it.
    skew > 0 replaces p with SRKL's (1 - skew) * p + skew * q, giving SRKL + DRKL (Figure 4a).
    """
    q_m, p_m, log_q_hat, log_p_hat = target_split(logits, teacher_logits, labels)
    if skew > 0:
        p_m = (1 - skew) * p_m + skew * q_m
        log_p_hat = torch.logaddexp(log_p_hat + math.log(1 - skew), log_q_hat + math.log(skew))
    return (_binary_kl(q_m, p_m) + gamma * _kl(log_q_hat, log_p_hat)).mean()


@torch.no_grad()
def rkl_terms(logits, teacher_logits, labels):
    """Batch means of the RKL decomposition (Figure 3) and of the non-target gradient on the
    target logit, d NRKL / d z_m = -q_m (1 - q_m) KL(q^ || p^) (Proposition 2, Figure 9)."""
    q_m, p_m, log_q_hat, log_p_hat = target_split(logits, teacher_logits, labels)
    non_target = _kl(log_q_hat, log_p_hat)
    q = q_m.squeeze(1)
    return {"q_m": q.mean().item(), "target_kl": _binary_kl(q_m, p_m).mean().item(),
            "non_target_kl": non_target.mean().item(), "grad_nrkl": (-q * (1 - q) * non_target).mean().item()}


def drkl_gamma(args, progress):
    """gamma, or DRKL-linear's gamma growing from --drkl-gamma-start to --drkl-gamma (Appendix F.5)."""
    if args.drkl_gamma_start is None:
        return args.drkl_gamma
    return args.drkl_gamma_start + (args.drkl_gamma - args.drkl_gamma_start) * progress


def distillation_loss(args, logits, teacher_logits, labels, progress=0.0):
    """The loss selected by --loss; progress in [0, 1] is the fraction of training done."""
    if args.loss == "fkl":
        return forward_kl(logits, teacher_logits, labels)
    if args.loss == "rkl":
        return reverse_kl(logits, teacher_logits, labels)
    if args.loss == "symkl":
        return symmetric_kl(logits, teacher_logits, labels)
    if args.loss == "dkd":
        return decoupled_kd(logits, teacher_logits, labels)
    if args.loss == "js":
        return js_divergence(logits, teacher_logits, labels)
    if args.loss == "sfkl":
        return skewed_forward_kl(logits, teacher_logits, labels, lam=args.skew_alpha)
    if args.loss == "srkl":
        return skewed_reverse_kl(logits, teacher_logits, labels, lam=args.skew_alpha)
    if args.loss == "akl":
        return adaptive_kl(logits, teacher_logits, labels)
    if args.loss == "ab":
        return ab_divergence(logits, teacher_logits, labels)
    if args.loss == "drkl":
        return diversity_aware_reverse_kl(logits, teacher_logits, labels, drkl_gamma(args, progress))
    if args.loss == "srkl_drkl":
        return diversity_aware_reverse_kl(logits, teacher_logits, labels, drkl_gamma(args, progress), args.skew_alpha)
    raise ValueError(f"unknown loss: {args.loss}")
