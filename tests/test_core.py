"""CPU checks of the losses, metrics, data pipeline and on-policy sampling: python -m pytest tests"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_utils.lm_datasets import LMTrainDataset  # noqa: E402
from data_utils.mmap_dataset import MMapDatasetBuilder  # noqa: E402
from data_utils.prompt_datasets import PromptDataset  # noqa: E402
from distillm import LOSSES, ReplayBuffer, SampleGenerator, distillation_loss, rkl_terms  # noqa: E402
from distillm.losses import _binary_kl, _kl, reverse_kl, target_split  # noqa: E402
from metrics import distinct_2, diversity_metrics, neg_self_bleu  # noqa: E402

KD_LOSSES = [loss for loss in LOSSES if loss != "sft"]
ARGS = SimpleNamespace(skew_alpha=0.1, drkl_gamma=1.5, drkl_gamma_start=None)


def make_logits(seed=0, shape=(2, 6, 11)):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, shape[-1], shape[:2], generator=g)
    labels[0, :2] = -100
    return torch.randn(shape, generator=g, dtype=torch.float64) * 2, labels


@pytest.mark.parametrize("loss", KD_LOSSES)
def test_gradient_vanishes_when_student_matches_teacher(loss):
    teacher, labels = make_logits()
    # a tiny offset: AKL's head/tail ratio is 0/0 at exact equality
    student = (teacher + 1e-6 * make_logits(1)[0]).requires_grad_()
    distillation_loss(SimpleNamespace(loss=loss, **vars(ARGS)), student, teacher, labels).backward()
    assert student.grad.abs().max() < 1e-5


@pytest.mark.parametrize("loss", KD_LOSSES)
def test_ignored_positions_do_not_matter(loss):
    student, labels = make_logits(0)
    teacher, _ = make_logits(1)
    args = SimpleNamespace(loss=loss, **vars(ARGS))
    perturbed = student.clone()
    perturbed[labels == -100] += 5.0
    assert torch.allclose(distillation_loss(args, student, teacher, labels),
                          distillation_loss(args, perturbed, teacher, labels))


def test_propositions_1_and_2():
    student, labels = make_logits(0)
    teacher, _ = make_logits(1)
    student.requires_grad_()
    q_m, p_m, log_q_hat, log_p_hat = target_split(student, teacher, labels)
    q_m = q_m.squeeze(1)
    target_kl, non_target_kl = _binary_kl(q_m.unsqueeze(1), p_m), _kl(log_q_hat, log_p_hat)
    # Proposition 1: RKL = KL(q~_m || p~_m) + (1 - q_m) KL(q^ || p^)
    assert torch.allclose(reverse_kl(student, teacher, labels), (target_kl + (1 - q_m) * non_target_kl).mean(), atol=1e-6)

    def grad_on_target_logit(term):
        grad = torch.autograd.grad(term.sum(), student, retain_graph=True)[0][labels != -100]
        return grad.gather(1, labels[labels != -100].unsqueeze(1)).squeeze(1).float()  # the losses compute in fp32

    # Proposition 2: RKL's non-target term always pushes the target logit up ...
    expected = -q_m * (1 - q_m) * non_target_kl
    assert torch.allclose(grad_on_target_logit((1 - q_m) * non_target_kl), expected, atol=1e-6)
    assert (expected < 0).all()
    assert rkl_terms(student, teacher, labels)["grad_nrkl"] == pytest.approx(expected.mean().item(), abs=1e-6)
    # ... while DRKL's constant weight removes that pathway
    assert torch.allclose(grad_on_target_logit(1.5 * non_target_kl), torch.zeros_like(q_m))


def test_diversity_metrics():
    assert distinct_2(["a b c", "a b d"]) == 0.75
    same = ["the cat sat on the mat"] * 3
    assert neg_self_bleu(same) == pytest.approx(-1.0)
    assert neg_self_bleu(["the cat sat on the mat", "a dog ran in a park"]) > -0.2
    assert diversity_metrics(same + ["a dog ran in a park"], group_size=3)["neg_self_bleu"] == pytest.approx(-0.5)


def test_dataset_masks_prompt_and_left_pads_generation_prompt(tmp_path):
    examples = [([1, 2, 3], [4, 5]), ([6], [7, 8, 9, 10])]
    builder = MMapDatasetBuilder(str(tmp_path / "train_0"), np.uint16)
    for prompt, response in examples:
        builder.add_item(prompt + [-1] + response)
    builder.finalize()
    (tmp_path / "train.jsonl").write_text("".join(json.dumps({"output": "x"}) + "\n" for _ in examples))

    args = SimpleNamespace(data_dir=str(tmp_path), max_length=16, max_prompt_length=4)
    data = LMTrainDataset(args, SimpleNamespace(eos_token_id=0), "train")
    model_batch, labels, gen_batch = data.collate([data[0], data[1]])
    assert model_batch["input_ids"][0, :4].tolist() == [1, 2, 3, 4]
    assert labels[0, :4].tolist() == [-100, -100, 4, 5]
    assert labels[1, :4].tolist() == [7, 8, 9, 10]
    assert gen_batch["input_ids"].tolist() == [[0, 1, 2, 3], [0, 0, 0, 6]]
    assert gen_batch["attention_mask"].tolist() == [[0, 1, 1, 1], [0, 0, 0, 1]]


def test_prompt_dataset_labels_the_reference_answer(tmp_path):
    rows = [{"prompt": "p p", "output": "a"}, {"prompt": "p", "output": ["b b", "c"]}]
    (tmp_path / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    vocab = {"p": 1, "a": 2, "b": 3, "c": 4}  # one token per word, eos = 9
    tokenizer = SimpleNamespace(eos_token_id=9, encode=lambda s, add_special_tokens: [vocab[w] for w in s.split()])
    args = SimpleNamespace(model_type="gpt2", max_length=8, max_prompt_length=3)
    data = PromptDataset(args, tokenizer, str(tmp_path))
    gen_batch, answer_batch, labels = data.collate([data[0], data[1]])
    assert gen_batch["input_ids"].tolist() == [[9, 1, 1], [9, 9, 1]]
    assert answer_batch["input_ids"].tolist() == [[1, 1, 2, 9], [1, 3, 3, 9]]
    assert labels.tolist() == [[-100, 2, 9, -100], [3, 3, 9, -100]]


def test_student_samples_are_labelled_with_the_next_token():
    pad = 0

    class StubModel:
        def eval(self):
            pass

        def generate(self, input_ids, attention_mask, **kwargs):
            responses = torch.tensor([[11, 12, pad], [13, pad, pad]])
            return torch.cat([input_ids, responses], dim=1)

    args = SimpleNamespace(max_length=8, max_prompt_length=3, student_gen_max_new_tokens=None)
    generator = SampleGenerator(args, SimpleNamespace(eos_token_id=pad, pad_token_id=pad))
    gen_batch = {"input_ids": torch.tensor([[pad, 1, 2], [3, 4, 5]]), "attention_mask": torch.ones(2, 3)}
    model_batch, labels = generator.run_sample(StubModel(), gen_batch)
    assert model_batch["input_ids"].tolist() == [[1, 2, 11, 12], [3, 4, 5, 13]]
    assert labels.tolist() == [[-100, 11, 12, -100], [-100, -100, 13, -100]]

    buffer = ReplayBuffer(capacity=4, batch_size=2, pad_id=pad)
    buffer.add({"input_ids": model_batch["input_ids"][:1, :3], "attention_mask": model_batch["attention_mask"][:1, :3]},
               labels[:1, :3])
    buffer.add(model_batch, labels)
    replay_batch, replay_labels = buffer.sample("cpu")
    assert replay_batch["input_ids"].shape == replay_labels.shape == (2, 4)


@pytest.mark.parametrize("text, rating", [("Rating: [[5]]", 5.0), ("[[rating: 7]]", 7.0),
                                          ("[[rating]]8[[/rating]]", 8.0), ("[[ 6.5 ]]", 6.5), ("no rating", None)])
def test_judge_reads_the_rating(text, rating):
    pytest.importorskip("anthropic")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from llm_judge import RATING
    match = RATING.search(text)
    assert (float(match.group(1) or match.group(2)) if match else None) == rating
