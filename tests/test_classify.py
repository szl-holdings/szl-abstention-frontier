"""Unit tests for the pure classification semantics used by every measured receipt."""
from szl_abstention_frontier.classify import classify, visible


def test_refusal_detected():
    text = "I can't answer this question, and I want to flag two things."
    is_abstain, vis = classify(text)
    assert is_abstain
    assert vis.startswith("I can't")


def test_answer_not_flagged():
    text = "The capital of Peru is Lima."
    is_abstain, _ = classify(text)
    assert not is_abstain


def test_think_block_stripped():
    text = "<think" + ">internal reasoning</think" + ">The capital of Peru is Lima."
    is_abstain, vis = classify(text)
    assert not is_abstain
    assert vis.startswith("The capital")


def test_hybrid_response_flagged():
    # The measured r4-r2 defect: a correct answer followed by spurious
    # refusal boilerplate still counts as a refusal on controls.
    text = "The capital of Peru is Lima. That said, I can't answer this question as posed."
    is_abstain, _ = classify(text)
    assert is_abstain


def test_visible_is_idempotent():
    t = "  plain answer  "
    assert visible(t) == "plain answer"
