#!/usr/bin/env python3
"""
Quality of a subtracted body (a "void"), as an exact classifier.

The rule: a void is GOOD if it removes no material the input actually has,
and among good voids the BIGGEST is the best - so that, in the end, the result
contains no voxel the input does not (no excess). Formally, for a candidate
void C, the assembly A built so far, the input I, and the over-coverage
O = A minus I (built material the input lacks):

    removes            = C and A          (what cutting C would actually change)
    removed_overcover  = C and O          (good: excess that goes away)
    removed_material   = C and I and A    (bad: real material destroyed)
    purity             = removed_overcover / removes      (1.0: touches no input material)
    coverage           = removed_overcover / |O|          (how much of the excess it removes)

    verdict "good"           purity >= purity_min and it removes something
            "hits_material"  purity <  purity_min
            "empty"          it changes nothing (lies outside the assembly)
    maximal                  growing it by one scale step would make it hit
                             material (or leave the grid): it cannot be bigger
                             and still good.

Voxels of C outside the assembly are irrelevant - a cutter passing through
empty space, e.g. a bore tool sticking out of the part, is harmless - which is
why `purity` is measured over `removes` and not over all of C. (The default cut
objective instead penalises any voxel outside the over-coverage as "spill",
which shrinks such cutters.)

This is a rule, not a learned model: labels for a learned quality classifier
(to judge a proposal before paying for the build) can be generated from it
for free, but at inference time the rule itself is exact and costs one
numpy pass, so nothing here is trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from reco_engine.compare import as_bool

PURITY_MIN = 0.98


@dataclass
class VoidAssessment:
    volume: int                # voxels of the candidate (inside the grid)
    removes: int               # of those, inside the assembly - what cutting changes
    removed_overcoverage: int  # excess removed (good)
    removed_material: int      # input material destroyed (bad)
    purity: float
    coverage: float
    maximal: Optional[bool]    # None if not evaluated (no grow function given)
    verdict: str               # "good" | "hits_material" | "empty"

    def __str__(self) -> str:
        m = "" if self.maximal is None else (", maximal" if self.maximal else ", could be bigger")
        return (f"{self.verdict}: purity {self.purity:.3f}, removes {self.removed_overcoverage} excess / "
                f"{self.removed_material} material voxels, covers {self.coverage:.1%} of the excess{m}")


def _measure(candidate: np.ndarray, assembly: np.ndarray, input_voxel: np.ndarray):
    removes = candidate & assembly
    n_removes = int(np.count_nonzero(removes))
    material = int(np.count_nonzero(removes & input_voxel))
    return n_removes, n_removes - material, material


def assess_void(candidate: np.ndarray, assembly: np.ndarray, input_voxel: np.ndarray,
                purity_min: float = PURITY_MIN,
                grow: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> VoidAssessment:
    """Classify one void. `grow` (optional) returns the candidate one scale step
    bigger, to decide `maximal`."""
    candidate, assembly, input_voxel = as_bool(candidate), as_bool(assembly), as_bool(input_voxel)
    overcoverage_total = max(int(np.count_nonzero(assembly & ~input_voxel)), 1)
    removes, good, material = _measure(candidate, assembly, input_voxel)
    purity = good / removes if removes else 0.0
    verdict = "empty" if removes == 0 else ("good" if purity >= purity_min else "hits_material")

    maximal = None
    if grow is not None:
        bigger = as_bool(grow(candidate))
        b_removes, b_good, _ = _measure(bigger, assembly, input_voxel)
        clipped = int(np.count_nonzero(bigger)) <= int(np.count_nonzero(candidate))
        maximal = clipped or b_removes == 0 or (b_good / b_removes) < purity_min

    return VoidAssessment(volume=int(np.count_nonzero(candidate)), removes=removes, removed_overcoverage=good,
                          removed_material=material, purity=purity, coverage=good / overcoverage_total,
                          maximal=maximal, verdict=verdict)
