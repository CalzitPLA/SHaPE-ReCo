"""Asking the user (reco_engine/advice.py): the protocol (default after the timeout, scripted answers, the record), the
hardness rules on fake candidates, the budget. No Qt."""
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import advice  # noqa: E402
from reco_engine.advice import (AdviceConfig, Answer, Budget, Candidate, NoAdviser, Question, QueueAdviser,  # noqa: E402
                                ScriptedAdviser, hardness)
from reco_engine.result_schema import Primitive  # noqa: E402

PIECE = dict(id=7, side="invented", op="cut", volume=120.0, share=0.02, bbox=((0, 0, 0), (1, 2, 3)))


def cand(kind, score, precision=None, recall=None, reason="", op="cut"):
    return dict(kind=kind, op=op, score=score, precision=precision, recall=recall, reason=reason, source="fit",
                primitive=Primitive(kind, dims_mm={"radius": 1.0}))


def question(qid="q1", timeout=60.0, default="skip", kind="close"):
    return Question(id=qid, kind=kind, text="which one?", piece=dict(PIECE),
                    candidates=[Candidate("cut cylinder", "cylinder", "cut", 0.8), Candidate("cut box", "box", "cut", 0.7)],
                    default=default, timeout_s=timeout)


def test_question_and_answer_are_plain_data():
    q = question()
    d = json.loads(json.dumps(q.as_dict()))
    back = Question.from_dict(d)
    assert back.as_dict() == d and back.candidates[0].score == 0.8 and back.default == "skip"
    a = Answer("q1", "candidate", {"index": 1}, "the box", "user")
    assert Answer.from_dict(json.loads(json.dumps(a.as_dict()))) == a
    assert set(advice.CHOICES) == {"candidate", "tool", "adjusted", "skip", "skip_group", "voxels", "leave", "stop", "default"}


def test_no_adviser_takes_the_default_at_once():
    adviser = NoAdviser()
    t0 = time.time()
    a = adviser.ask(question(timeout=30.0))
    assert time.time() - t0 < 0.1 and a.choice == "skip" and a.source == "default" and a.question == "q1"
    a = adviser.ask(Question("q2", "ambiguous", "?", default="candidate", default_parameters={"index": 0}))
    assert a.choice == "candidate" and a.parameters == {"index": 0}
    assert [r["question"]["id"] for r in adviser.record] == ["q1", "q2"]


def test_scripted_adviser_answers_from_a_file(tmp_path):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(dict(answers=[
        dict(kind="ambiguous", choice="candidate", parameters={"index": 1}, note="by kind"),
        dict(question="q3", choice="stop"),
        dict(choice="voxels", note="the next one"),
        dict(question="q9", choice="dance")])))
    adviser = ScriptedAdviser(path)
    a1 = adviser.ask(question("q1", kind="close"))                  # no id, no kind fits: the entry without both
    assert (a1.choice, a1.source, a1.note) == ("voxels", "scripted", "the next one")
    a2 = adviser.ask(question("q2", kind="ambiguous"))
    assert a2.choice == "candidate" and a2.parameters == {"index": 1} and a2.source == "scripted"
    assert adviser.ask(question("q3", kind="ambiguous")).choice == "stop"              # by its id
    a4 = adviser.ask(question("q4"))                                 # nothing left for it: the default
    assert a4.choice == "skip" and a4.source == "default" and a4.note == "no scripted answer"
    a9 = adviser.ask(question("q9"))                                 # an answer that is none
    assert a9.choice == "skip" and "'dance' is no answer" in a9.note
    assert len(adviser.record) == 5 and ScriptedAdviser([dict(choice="leave")]).ask(question()).choice == "leave"
    limited = Question("q", "close", "?", default="skip", choices=("candidate", "skip"))
    assert ScriptedAdviser([dict(choice="voxels")]).ask(limited).choice == "skip"       # not offered for this question


def test_queue_adviser_default_after_the_timeout():
    seen = []
    adviser = QueueAdviser(on_question=seen.append)
    t0 = time.time()
    a = adviser.ask(question(timeout=0.3))
    took = time.time() - t0
    assert 0.3 <= took < 1.0 and a.choice == "skip" and a.source == "default after timeout"
    assert [q.id for q in seen] == ["q1"] and adviser.waiting is None
    r = adviser.record[0]
    assert r["answer"]["source"] == "default after timeout" and 0.3 <= r["seconds"] < 1.0 and r["asked"] <= r["answered"]
    assert QueueAdviser(timeout_s=0.1).ask(question(timeout=600.0)).source == "default after timeout"


def test_queue_adviser_is_answered_from_another_thread():
    adviser = QueueAdviser()
    got = {}
    worker = threading.Thread(target=lambda: got.update(answer=adviser.ask(question(timeout=20.0))))
    t0 = time.time()
    worker.start()
    while adviser.waiting is None and time.time() - t0 < 5:
        time.sleep(0.01)
    assert adviser.waiting.id == "q1" and worker.is_alive()
    adviser.answer(Answer("other", "stop"))                          # not this question: it goes on waiting
    time.sleep(0.1)
    assert worker.is_alive()
    adviser.answer(Answer("q1", "candidate", {"index": 1}, "the box"))
    worker.join(5)
    assert not worker.is_alive() and time.time() - t0 < 5
    assert got["answer"].choice == "candidate" and got["answer"].source == "user" and got["answer"].parameters == {"index": 1}
    # an answer given before the question is asked (the GUI saw the "advice_asked" step): at once
    adviser.answer(Answer("q2", "leave"))
    t0 = time.time()
    assert adviser.ask(question("q2", timeout=20.0)).choice == "leave" and time.time() - t0 < 0.1
    assert [r["question"]["id"] for r in advice.record_of(adviser)] == ["q1", "q2"]
    # closing: the waiting question gets its default
    late = threading.Thread(target=lambda: got.update(late=adviser.ask(question("q3", timeout=20.0))))
    late.start()
    while adviser.waiting is None:
        time.sleep(0.01)
    adviser.close()
    late.join(5)
    assert got["late"].choice == "skip" and got["late"].source == "default"
    json.dumps(advice.record_of(adviser))


def test_hardness_rules():
    cfg = AdviceConfig()
    assert (cfg.close_margin, cfg.scoring_error, cfg.group_min_copies, cfg.give_up_share, cfg.min_share,
            cfg.max_questions, cfg.max_per_minute, cfg.timeout_s) == (0.10, 0.03, 3, 0.01, 0.001, 5, 2, 60.0)
    thr = 0.85
    # not hard: a clear winner, a clear failure, a small piece
    assert hardness(PIECE, [cand("cylinder", 0.95), cand("box", 0.5)], thr, accepted=True) is None
    assert hardness(PIECE, [cand("cylinder", 0.4), cand("box", 0.3)], thr) is None
    assert hardness(dict(PIECE, share=0.0005), [cand("cylinder", 0.84)], thr, giving_up=True, copies=9) is None
    # close: fails, by at most the margin
    q = hardness(PIECE, [cand("box", 0.5), cand("cylinder", 0.80, 0.8, 0.95, "best score 0.800 < 0.85")], thr, question_id="q1")
    assert q.kind == "close" and q.id == "q1" and q.default == "skip" and q.timeout_s == 60.0
    assert [c.kind for c in q.candidates] == ["cylinder", "box"] and q.candidates[0].precision == 0.8
    assert q.candidates[0].primitive["kind"] == "cylinder" and "0.800" in q.text and "piece 7" in q.text
    assert q.piece["id"] == 7 and "skip_group" not in q.choices and "candidate" in q.choices
    assert hardness(PIECE, [cand("cylinder", 0.74)], thr) is None                       # 0.11 below: not close
    assert hardness(PIECE, [cand("cylinder", 0.75)], thr).kind == "close"
    assert hardness(PIECE, [cand("cylinder", 0.90, precision=0.9)], thr, precision_threshold=0.95).kind == "close"
    assert hardness(PIECE, [cand("cylinder", 0.90, precision=0.8)], thr, precision_threshold=0.95) is None
    # ambiguous: two kinds within the scoring error (also when one was accepted)
    q = hardness(PIECE, [cand("cylinder", 0.91), cand("cone", 0.89)], thr, accepted=True)
    assert q.kind == "ambiguous" and q.default == "candidate" and q.default_parameters == {"index": 0}
    assert hardness(PIECE, [cand("cylinder", 0.91), cand("cylinder", 0.90)], thr, accepted=True) is None   # the same kind
    assert hardness(PIECE, [cand("cylinder", 0.91), cand("cone", 0.87)], thr, accepted=True) is None      # 0.04 apart
    assert hardness(PIECE, [cand("cylinder", 0.5), cand("cone", 0.49)], thr) is None                       # both far off
    # group: the representative failed and blocks copies
    q = hardness(PIECE, [cand("box", 0.3, reason="no fit")], thr, copies=3)
    assert q.kind == "group" and q.default == "skip_group" and "3 copies" in q.text and "skip_group" in q.choices
    assert hardness(PIECE, [cand("box", 0.3)], thr, copies=2) is None
    # give up: a piece of at least 1 % of the part
    q = hardness(PIECE, [], thr, giving_up=True)
    assert q.kind == "give_up" and q.default == "leave" and "candidate" not in q.choices and q.candidates == []
    assert hardness(dict(PIECE, share=0.009), [], thr, giving_up=True) is None
    assert hardness(PIECE, [cand("box", 0.3)], thr, giving_up=True, accepted=True) is None
    # the thresholds come from the config
    wide = AdviceConfig(close_margin=0.3, max_candidates=1, timeout_s=5.0)
    q = hardness(PIECE, [cand("cylinder", 0.6), cand("box", 0.58)], thr, wide)
    assert q.kind == "close" and len(q.candidates) == 1 and q.timeout_s == 5.0


def test_budget_does_not_pester():
    now = [0.0]
    budget = Budget(AdviceConfig(max_questions=5, max_per_minute=2), clock=lambda: now[0])
    assert budget.left() == 5 and budget.spend() and budget.spend()
    assert not budget.allows() and not budget.spend()                 # two in this minute
    now[0] = 61.0
    assert budget.spend() and budget.spend() and not budget.spend()
    now[0] = 200.0
    assert budget.spend() and budget.left() == 0
    now[0] = 1000.0
    assert not budget.allows() and not budget.spend()                 # five in the run
    qs = [hardness(dict(PIECE, id=i, share=s), [], 0.85, giving_up=True, question_id=f"q{i}")
          for i, s in enumerate((0.02, 0.3, 0.05, 0.011, 0.2, 0.1, 0.04))]
    chosen = Budget(AdviceConfig(max_questions=3)).choose(qs)
    assert [q.piece["share"] for q in chosen] == [0.3, 0.2, 0.1]      # the largest pieces first
