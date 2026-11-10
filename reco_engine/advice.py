#!/usr/bin/env python3
"""
Asking the user for advice when a guess is hard.

No Qt here. `Question` and `Answer` are the protocol, an `Adviser` answers questions:

    NoAdviser        always the default, at once: every scripted and headless run behaves as without this module
    ScriptedAdviser  the answers of a JSON file (tests, reproducible runs)
    QueueAdviser     thread-safe: the loop's worker thread blocks in `ask` until another thread (the GUI's) calls
                     `answer`, or the question's timeout takes the default

Every question and its answer are recorded with their times (`Adviser.record`, a list of dicts: what goes into the
run's steps and into run.json as `advice: [...]`), so a result shows where a human decided.

WHEN IS A GUESS HARD? One place: `hardness(...)`, thresholds in `AdviceConfig` (all documented there):
    "close"      the best candidate fails the acceptance, but by at most `close_margin` (score or precision)
    "ambiguous"  the two best candidates are of different kinds and within `scoring_error` of each other
    "group"      a group's representative failed and its failure blocks >= `group_min_copies` copies
    "give_up"    the phase is about to give up a piece of >= `give_up_share` of the part's volume
A `Budget` keeps the framework from pestering: at most `max_questions` per run (5) and `max_per_minute` (2), no piece
below `min_share` of the part, and among several hard pieces the largest first (`Budget.choose`).

THE CONTRACT WITH THE LOOP (reco_engine/residual_loop.py implements it; the GUI passes the keywords only when the
config has the fields):

  ResidualConfig.adviser: an Adviser, default NoAdviser() (None means NoAdviser).
  ResidualConfig.advice:  an AdviceConfig, default AdviceConfig() (the thresholds and the budget).
  Where the loop decides about a piece / a component / a group, it calls
        q = advice.hardness(piece, candidates, threshold, cfg.advice, ...)      # None: not hard, go on as today
        if q is not None and budget.allows(q):
            yield ResidualStep("advice_asked", snapshot, dict(question=q.as_dict(), piece_id=..., mesh=<the piece's
                               mesh: dict(vertices, faces), as in "feature_piece">, candidates=[Primitive, ...] in the
                               order of q.candidates, message=q.text, phase=<the phase>))
            a = cfg.adviser.ask(q)                                                # blocks; never raises
            yield ResidualStep("advice_given", snapshot, dict(answer=a.as_dict(), question_id=q.id,
                               source=a.source, message=<one line>, phase=<the phase>))
    "advice_asked" is yielded BEFORE `ask` is called: a supervisor sees the question (the GUI stops the run at this
    step, shows it and puts the answer into its QueueAdviser) and `ask` then returns at once.
  Answer.source is "user", "default after timeout", "default" (NoAdviser, or the budget was used up) or "scripted".
  What the loop does with Answer.choice (one of CHOICES):
    "candidate"   take candidates[parameters["index"]] as if it had won; it is still validated (the FreeCAD build,
                  the exact residual) and may be rejected - the verdict step says so
    "tool"        guess the piece again with the tool class parameters["class"] only (reco_engine.tools)
    "adjusted"    parameters["shape"] is a user_guess.UserShape dict: refine it (user_guess.refine) and evaluate it
                  like a first guess
    "skip"        give this piece up (a "feature_rejected" / "rejected" with reason "skipped by the user")
    "skip_group"  ... and every piece of its group
    "voxels"      explain the piece by the voxel-cube fallback
    "leave"       leave the piece in the residual and do not ask about it again
    "stop"        end the phase (stopped "stopped by the user")
    "default"     what the algorithm does by itself (the question's `default` choice)
  An answer the loop cannot use (an index out of range, an unknown class) counts as "default" and the
  "advice_given" step says so in its message.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

CHOICES = ("candidate", "tool", "adjusted", "skip", "skip_group", "voxels", "leave", "stop", "default")
KINDS = ("close", "ambiguous", "group", "give_up")
SOURCES = ("user", "default after timeout", "default", "scripted")


@dataclass
class Candidate:
    """One option of a question: a candidate of the loop with its numbers and a short reason."""
    label: str                                      # "cut cylinder r 4.0"
    kind: str = ""                                  # the primitive's kind (or the tool class)
    op: str = ""
    score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    reason: str = ""                                # why it failed / what speaks for it
    source: Optional[str] = None
    primitive: Optional[dict] = None                # its result_schema.Primitive as a dict (dataclasses.asdict)

    def as_dict(self) -> dict:
        return dict(label=self.label, kind=self.kind, op=self.op, score=self.score, precision=self.precision,
                    recall=self.recall, reason=self.reason, source=self.source, primitive=self.primitive)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        return cls(**{k: d.get(k) for k in ("label", "kind", "op", "score", "precision", "recall", "reason", "source",
                                            "primitive") if d.get(k) is not None or k in ("score", "precision", "recall")})


@dataclass
class Question:
    id: str                                         # unique in the run, e.g. "q3"
    kind: str                                       # one of KINDS
    text: str                                       # one readable sentence
    piece: Optional[dict] = None                    # the piece / component: id, side, op, volume, share, bbox, copies
    candidates: List[Candidate] = field(default_factory=list)
    default: str = "default"                        # the choice the algorithm takes by itself (one of CHOICES)
    default_parameters: Dict[str, object] = field(default_factory=dict)
    timeout_s: float = 60.0                         # after this long the default is taken
    choices: Sequence[str] = CHOICES                # the answers that make sense for this question

    def as_dict(self) -> dict:
        return dict(id=self.id, kind=self.kind, text=self.text, piece=self.piece,
                    candidates=[c.as_dict() for c in self.candidates], default=self.default,
                    default_parameters=dict(self.default_parameters), timeout_s=self.timeout_s, choices=list(self.choices))

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(id=str(d.get("id", "")), kind=str(d.get("kind", "")), text=str(d.get("text", "")), piece=d.get("piece"),
                   candidates=[Candidate.from_dict(c) for c in d.get("candidates") or () if isinstance(c, dict)],
                   default=str(d.get("default") or "default"), default_parameters=dict(d.get("default_parameters") or {}),
                   timeout_s=float(d.get("timeout_s") if d.get("timeout_s") is not None else 60.0),
                   choices=tuple(d.get("choices") or CHOICES))

    def default_answer(self, source: str = "default", note: str = "") -> "Answer":
        return Answer(self.id, self.default, dict(self.default_parameters), note, source)


@dataclass
class Answer:
    question: str                                   # the question's id
    choice: str                                     # one of CHOICES
    parameters: Dict[str, object] = field(default_factory=dict)      # "candidate": index; "tool": class; "adjusted": shape
    note: str = ""
    source: str = "user"                            # one of SOURCES

    def as_dict(self) -> dict:
        return dict(question=self.question, choice=self.choice, parameters=dict(self.parameters), note=self.note,
                    source=self.source)

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        return cls(question=str(d.get("question", "")), choice=str(d.get("choice") or "default"),
                   parameters=dict(d.get("parameters") or {}), note=str(d.get("note") or ""),
                   source=str(d.get("source") or "user"))


# ------------------------------------------------------------------------------------------------ advisers

class Adviser:
    """`ask(question) -> Answer`; never raises. `record`: every question with its answer and their times."""

    def __init__(self):
        self.record: List[dict] = []
        self._lock = threading.Lock()

    def ask(self, question: Question) -> Answer:
        t0 = time.time()
        try:
            answer = self._ask(question)
        except Exception as exc:                                    # an adviser that fails is the default
            answer = question.default_answer("default", f"the adviser failed: {str(exc)[-200:]}")
        if answer.choice not in CHOICES or answer.choice not in tuple(question.choices) + ("default",):
            answer = question.default_answer(answer.source, f"'{answer.choice}' is no answer to this question")
        if answer.choice == "default":                              # say what the default is
            answer = Answer(question.id, question.default, dict(question.default_parameters), answer.note, answer.source)
        with self._lock:
            self.record.append(dict(question=question.as_dict(), answer=answer.as_dict(), asked=t0, answered=time.time(),
                                    seconds=time.time() - t0))
        return answer

    def _ask(self, question: Question) -> Answer:
        raise NotImplementedError


class NoAdviser(Adviser):
    """Every question gets its default, at once."""

    def _ask(self, question: Question) -> Answer:
        return question.default_answer("default")


class ScriptedAdviser(Adviser):
    """The answers of a JSON file (or a list of dicts): {"answers": [{"question": id or null, "kind": kind or null,
    "choice": ..., "parameters": {...}, "note": ...}, ...]}. An entry answers the first question it fits - by its id,
    else by its kind, else (neither given) the next question - and is used once; a question without an entry gets its
    default."""

    def __init__(self, answers):
        super().__init__()
        if isinstance(answers, (str, Path)):
            data = json.loads(Path(answers).read_text())
            answers = data.get("answers", data) if isinstance(data, dict) else data
        self.answers = [dict(a) for a in answers]

    def _ask(self, question: Question) -> Answer:
        for rule in ((lambda a: a.get("question") == question.id),
                     (lambda a: not a.get("question") and a.get("kind") == question.kind),
                     (lambda a: not a.get("question") and not a.get("kind"))):
            for i, a in enumerate(self.answers):
                if rule(a):
                    del self.answers[i]
                    return Answer(question.id, str(a.get("choice") or "default"), dict(a.get("parameters") or {}),
                                  str(a.get("note") or ""), "scripted")
        return question.default_answer("default", "no scripted answer")


class QueueAdviser(Adviser):
    """Thread-safe: `ask` (the loop's worker thread) waits until `answer` is called from another thread, or until the
    question's timeout, which takes the default. An answer may arrive BEFORE `ask` (the GUI answers the question it
    saw in the "advice_asked" step): `ask` then returns at once. `on_question(question)` is called in the asking
    thread when it starts to wait."""

    def __init__(self, on_question: Optional[Callable[[Question], None]] = None, timeout_s: Optional[float] = None):
        super().__init__()
        self.on_question = on_question
        self.timeout_s = timeout_s                                   # None: each question's own
        self._answers: Dict[str, Answer] = {}
        self._condition = threading.Condition()
        self.waiting: Optional[Question] = None
        self._closed = False

    def answer(self, answer: Answer) -> None:
        with self._condition:
            self._answers[answer.question] = answer
            self._condition.notify_all()

    def close(self) -> None:
        """Every waiting and every later question gets its default (the window closes, the run is stopped)."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _ask(self, question: Question) -> Answer:
        limit = question.timeout_s if self.timeout_s is None else self.timeout_s
        end = time.time() + max(0.0, float(limit))
        with self._condition:
            if question.id not in self._answers and not self._closed:
                self.waiting = question
                if self.on_question is not None:
                    try:
                        self.on_question(question)
                    except Exception:
                        pass
                while question.id not in self._answers and not self._closed:
                    left = end - time.time()
                    if left <= 0 or not self._condition.wait(min(left, 0.25)):
                        if time.time() >= end:
                            break
            self.waiting = None
            given = self._answers.pop(question.id, None)
        if given is not None:
            return given
        return question.default_answer("default" if self._closed else "default after timeout")


# ------------------------------------------------------------------------------------------------ hardness

@dataclass
class AdviceConfig:
    """When a guess is hard, and how often the framework may ask."""
    close_margin: float = 0.10          # "close": the best candidate misses the threshold by at most this (score or precision)
    scoring_error: float = 0.03         # "ambiguous": two candidates this close in score are not told apart by the score
    group_min_copies: int = 3           # "group": a failed representative that blocks at least this many copies
    give_up_share: float = 0.01         # "give_up": a piece of at least this share of the part's volume
    min_share: float = 0.001            # no question about a piece smaller than this share of the part
    max_questions: int = 5              # per run
    max_per_minute: int = 2
    timeout_s: float = 60.0             # a question's default is taken after this long
    max_candidates: int = 4             # candidates shown with a question


def _value(c, name):
    return c.get(name) if isinstance(c, dict) else getattr(c, name, None)


def _candidate(c) -> Candidate:
    if isinstance(c, Candidate):
        return c
    p = _value(c, "primitive")
    kind = _value(c, "kind") or (_value(p, "kind") if p is not None else "") or ""
    op = _value(c, "op") or ""
    prim = None
    if p is not None:
        try:
            from dataclasses import asdict, is_dataclass
            prim = asdict(p) if is_dataclass(p) else dict(p)
        except Exception:
            prim = None
    numbers = {k: (float(_value(c, k)) if _value(c, k) is not None else None) for k in ("score", "precision", "recall")}
    return Candidate(label=str(_value(c, "label") or f"{op} {kind}".strip()), kind=str(kind), op=str(op),
                     reason=str(_value(c, "reason") or ""), source=_value(c, "source"), primitive=prim, **numbers)


def hardness(piece: Optional[dict], candidates: Sequence, threshold: float, config: Optional[AdviceConfig] = None,
             accepted: bool = False, copies: int = 0, giving_up: bool = False, precision_threshold: Optional[float] = None,
             question_id: str = "q") -> Optional[Question]:
    """The question to ask about a piece, or None when its guess is not hard.

    `piece`: dict with at least "share" (of the part's volume; also id, side, op, volume, bbox). `candidates`: the
    loop's candidates (dicts or objects with kind / op / score / precision / recall / reason / primitive), any order.
    `threshold`: the score a candidate needs. `accepted`: a candidate was accepted already (then only "ambiguous"
    can make it hard). `copies`: the copies of its group the piece's failure blocks. `giving_up`: the phase is about
    to give the piece up. The rules in their order (the first that holds names the question's kind):
      close, ambiguous, group, give_up - see the module docstring and AdviceConfig."""
    cfg = config or AdviceConfig()
    piece = dict(piece or {})
    share = float(piece.get("share") or 0.0)
    if share < cfg.min_share:
        return None
    ranked = sorted((_candidate(c) for c in candidates), key=lambda c: -(c.score if c.score is not None else -1.0))
    shown = ranked[:cfg.max_candidates]
    best = ranked[0] if ranked else None
    name = f"piece {piece.get('id', '?')} ({share:.1%} of the part)"
    kind = text = None
    default, parameters = ("default", {})
    if best is not None and best.score is not None and not accepted:
        missing = threshold - best.score
        short = (precision_threshold is not None and best.precision is not None
                 and 0 < precision_threshold - best.precision <= cfg.close_margin and best.score >= threshold)
        if 0 < missing <= cfg.close_margin or short:
            kind = "close"
            text = (f"{name}: the best guess, {best.label}, scores {best.score:.3f} - "
                    + (f"{missing:.3f} below the {threshold:g} it needs" if missing > 0 else
                       f"enough, but its precision {best.precision:.3f} is below {precision_threshold:g}")
                    + ". Take it, adjust it, or skip the piece?")
            default = "skip"
    if kind is None and len(ranked) >= 2 and ranked[0].score is not None and ranked[1].score is not None \
            and ranked[0].kind != ranked[1].kind and ranked[0].score - ranked[1].score <= cfg.scoring_error \
            and ranked[0].score >= threshold - cfg.close_margin:
        kind = "ambiguous"
        text = (f"{name}: {ranked[0].label} ({ranked[0].score:.3f}) and {ranked[1].label} ({ranked[1].score:.3f}) are "
                f"within the scoring error ({cfg.scoring_error:g}). Which one is it?")
        default, parameters = ("candidate", {"index": 0}) if ranked[0].score >= threshold else ("skip", {})
    if kind is None and not accepted and int(copies) >= cfg.group_min_copies:
        kind = "group"
        text = (f"{name}: its guess failed" + (f" ({best.reason})" if best is not None and best.reason else "")
                + f", and {int(copies)} copies of it wait for it. What is it?")
        default = "skip_group"
    if kind is None and giving_up and not accepted and share >= cfg.give_up_share:
        kind = "give_up"
        text = (f"{name}: nothing explains it" + (f" (best: {best.label}, {best.score:.3f})" if best is not None
                                                  and best.score is not None else "")
                + ". Skip it, cut it as voxels, or place a shape?")
        default = "leave"
    if kind is None:
        return None
    choices = [c for c in CHOICES if c != "skip_group" or int(copies) > 0]
    if not shown:
        choices = [c for c in choices if c != "candidate"]
    return Question(id=question_id, kind=kind, text=text, piece=dict(piece, copies=int(copies)), candidates=shown,
                    default=default, default_parameters=parameters, timeout_s=cfg.timeout_s, choices=tuple(choices))


class Budget:
    """How often the framework may ask: `max_questions` per run, `max_per_minute`, largest pieces first."""

    def __init__(self, config: Optional[AdviceConfig] = None, clock: Callable[[], float] = time.time):
        self.config = config or AdviceConfig()
        self.clock = clock
        self.asked: List[float] = []

    def left(self) -> int:
        return max(0, self.config.max_questions - len(self.asked))

    def allows(self, question: Optional[Question] = None) -> bool:
        """May a question be asked now? (Does not count it: `spend` does.)"""
        if self.left() <= 0:
            return False
        now = self.clock()
        return sum(1 for t in self.asked if now - t < 60.0) < self.config.max_per_minute

    def spend(self, question: Optional[Question] = None) -> bool:
        """Count a question if it may be asked; False when the budget does not allow it."""
        if not self.allows(question):
            return False
        self.asked.append(self.clock())
        return True

    def choose(self, questions: Sequence[Question]) -> List[Question]:
        """Of several hard pieces, the ones to ask about: the largest first, as many as the run's budget has left."""
        ranked = sorted(questions, key=lambda q: -float((q.piece or {}).get("share") or 0.0))
        return ranked[:self.left()]


def record_of(adviser) -> List[dict]:
    """The adviser's record as plain data (run.json's `advice`)."""
    return [json.loads(json.dumps(r, default=str)) for r in (getattr(adviser, "record", None) or [])]
