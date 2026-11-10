#!/usr/bin/env python3
"""
Background execution of the reconstruction. Loading the networks, the exact field, calling FreeCAD and the loop itself take
seconds, so they run in a worker thread; the GUI thread only draws finished states. The worker owns the StepSession:
the GUI asks it (through queued signals) to start, to advance one step, or to continue until a breakpoint, and
pauses it through a thread-safe event.
"""

from __future__ import annotations

import threading
import traceback

from PyQt5 import QtCore


class Worker(QtCore.QObject):
    stateReady = QtCore.pyqtSignal(int)          # index into session.states
    status = QtCore.pyqtSignal(str)
    started = QtCore.pyqtSignal(object)          # the new StepSession
    idle = QtCore.pyqtSignal(str)                # "ready" | "breakpoint" | "paused" | "limit" | "finished"
    failed = QtCore.pyqtSignal(str)
    busy = QtCore.pyqtSignal(bool)

    def __init__(self, make_session=None):
        super().__init__()
        self._make_session = make_session         # injectable for tests: (settings, models or None) -> (session, models, workdir)
        self.session = None
        self.settings = None
        self.workdir = None
        self.stop_event = threading.Event()

    # called from the GUI thread
    def pause(self):
        self.stop_event.set()

    @staticmethod
    def _in_python_thread(fn):
        """Run `fn()` in a thread Python created and return its result (an exception is re-raised here).

        Why: the slots below run in a QThread, whose Python thread state is SIP's. The networks (torch, pybind11)
        acquire the GIL through PyGILState, which binds its own thread state to the OS thread; on a later run of the
        same worker thread the two disagree and Python 3.13 aborts with "Fatal Python error: _PyThreadState_Attach:
        non-NULL old thread state" at reco_engine/predict.py (_type_tensor) - the GUI died whenever a second part was
        run after a first (the launcher log, 2026-10-10 and 2026-10-15). A thread from `threading` has a
        PyGILState-managed state from the start, so every torch call inside it is consistent; the QThread only waits
        (the GIL released in join), and the signals emitted from the inner thread are queued to the GUI as before."""
        box = {}

        def body():
            try:
                box["result"] = fn()
            except BaseException as exc:                              # carried over, raised in the caller
                box["error"] = exc
        t = threading.Thread(target=body, name="shape-reco-session", daemon=True)
        t.start()
        t.join()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    @QtCore.pyqtSlot(object)
    def start(self, settings):
        self.busy.emit(True)
        self.stop_event.clear()
        try:
            self.status.emit("Preparing the run ...")
            make = self._make_session
            if make is None:
                from reco_gui.setup import make_session as make
            session, _models, workdir = self._in_python_thread(lambda: make(settings, None))
            self.session, self.settings, self.workdir = session, settings, workdir
            self.started.emit(session)
            self.idle.emit("ready")
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.busy.emit(False)

    def _run(self, breakpoints, delay, max_steps):
        if self.session is None or self.session.finished:
            self.idle.emit("finished")
            return
        self.busy.emit(True)
        self.stop_event.clear()
        try:
            self.status.emit("Computing ...")
            session = self.session
            reason = self._in_python_thread(lambda: session.run(
                breakpoints=breakpoints, delay=delay, stop=self.stop_event,
                paced=getattr(session, "paced", None),                 # None: wait after every stage
                on_state=lambda st: self.stateReady.emit(st.index), max_steps=max_steps))
            self.idle.emit(reason)
        except Exception:
            self.failed.emit(self.session.error or traceback.format_exc())
        finally:
            self.busy.emit(False)

    @QtCore.pyqtSlot()
    def step(self):
        self._run(breakpoints=(), delay=0.0, max_steps=1)

    @QtCore.pyqtSlot(object, float)
    def cont(self, breakpoints, delay):
        self._run(breakpoints=set(breakpoints), delay=delay, max_steps=None)
