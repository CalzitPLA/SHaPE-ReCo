#!/usr/bin/env python3
"""
The Qt-free stepper of the supervision GUI: it advances an engine's step generator one stage at a time and folds every
yielded step into an immutable state - everything the front end draws after that step - so that any earlier step can
be shown again without re-running anything. The folds are the modes' own (reco_gui.cascade_session,
reco_gui.field_session); the voxel mode's fold (route C) was removed on 2026-09-30.

Pacing lives here too (`StepSession.run`): advance until a chosen stage is reached, the run finishes, or a stop
flag is set, optionally waiting between steps so that a human can follow. The GUI only calls these methods from a
worker thread and draws the states; nothing here imports Qt, torch or FreeCAD, so it is tested with a stub generator.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable, Iterator, List, Optional

# The stages `StepSession.run` waits after (its `delay`, the GUI's pace) when the caller passes them as `paced`: the
# steps a supervisor wants to look at. The vector-field mode waits after every stage (paced None); the cascade mode
# does not wait after its candidates (reco_gui.cascade_session.CASCADE_PACED_STAGES: on the liner about 330 candidate
# steps x the pace were 130 s of a 244 s run).
PACED_STAGES = None


class StepSession:
    """One reconstruction being stepped through. `states[i]` is the view after step i."""

    def __init__(self, steps: Iterator, fold: Callable):
        """`fold(previous_state or None, step, index) -> state` turns a step into a state: the cascade GUI passes
        reco_gui.cascade_session's folder, the vector-field GUI reco_gui.field_session.FieldFolder(). A state has at
        least `index`, `step` (with `stage`) and `finished`."""
        self._steps = steps
        self._fold = fold
        self.states: List = []
        self.finished = False
        self.error: Optional[str] = None

    @property
    def last(self) -> Optional[object]:
        return self.states[-1] if self.states else None

    def advance(self) -> Optional[object]:
        """Run the engine to its next yield. None once the run is over (or failed: see `error`)."""
        if self.finished:
            return None
        try:
            step = next(self._steps)
        except StopIteration:
            self.finished = True
            return None
        except Exception:
            self.finished = True
            self.error = traceback.format_exc()
            raise
        state = self._fold(self.last, step, len(self.states))
        self.states.append(state)
        if state.finished:
            self.finished = True
        return state

    def run(self, breakpoints=(), delay: float = 0.0, stop: Optional[threading.Event] = None,
            on_state: Optional[Callable[[object], None]] = None, max_steps: Optional[int] = None,
            paced=None) -> str:
        """Advance until the new state's stage is in `breakpoints` ("breakpoint"), the run is over ("finished"),
        `stop` is set ("paused") or `max_steps` states were produced ("limit"). `delay` seconds are waited after
        every state (interruptibly) so that a supervisor can follow what happens; with `paced` (a collection of
        stages) only after the states of those stages - the others are folded, shown and can be breakpoints, without
        a wait."""
        stop = stop or threading.Event()
        produced = 0
        while True:
            if stop.is_set():
                return "paused"
            state = self.advance()
            if state is None:
                return "finished"
            produced += 1
            if on_state is not None:
                on_state(state)
            if state.finished:
                return "finished"
            if state.step.stage in breakpoints:
                return "breakpoint"
            if max_steps is not None and produced >= max_steps:
                return "limit"
            if delay > 0 and (paced is None or state.step.stage in paced) and stop.wait(delay):
                return "paused"
