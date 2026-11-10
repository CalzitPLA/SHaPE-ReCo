# -*- coding: utf-8 -*-
"""
Shared setup for all shape_reco nodes: puts the Sandbox project root on
sys.path (so `import reco_engine`, `import training` work from inside
FreeCAD's Python) and caches loaded CNN models so each node evaluation
doesn't reload three checkpoints from disk.

Not a node itself - the Nodes workbench's loader execs every .py file
under nodes/, but a file with no @register_node class simply contributes
no nodes, so this is safe to sit alongside the real node modules.

The Sandbox root is found, in order, via the SHAPE_RECO_SANDBOX_ROOT
environment variable, the _sandbox_root.txt file that install.sh writes
next to the installed nodes, and (when run straight from the repo) this
file's own location.
"""

import os
import sys
from pathlib import Path


def _find_sandbox_root() -> Path:
    env = os.environ.get("SHAPE_RECO_SANDBOX_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    installed = here / "_sandbox_root.txt"
    if installed.is_file():
        return Path(installed.read_text().strip())
    return here.parent.parent  # <root>/freecad_nodes/shape_reco/_bootstrap.py


SANDBOX_ROOT = _find_sandbox_root()

if not (SANDBOX_ROOT / "reco_engine").is_dir():
    raise RuntimeError(
        f"shape_reco nodes: Sandbox project root not found at {SANDBOX_ROOT}. "
        f"Re-run freecad_nodes/install.sh or set the SHAPE_RECO_SANDBOX_ROOT "
        f"environment variable to the correct path."
    )

if str(SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(SANDBOX_ROOT))


from qtpy.QtWidgets import QPushButton, QVBoxLayout  # noqa: E402
from nodeeditor.node_content_widget import QDMNodeContentWidget  # noqa: E402


class RunButtonContent(QDMNodeContentWidget):
    """Content widget for nodes whose eval_operation is slow and/or
    side-effecting (shells out to freecadcmd, runs the CNNs, draws into
    the 3D view) and shouldn't fire the moment the graph is opened.

    The Nodes workbench evaluates every node unconditionally in three
    places outside anyone's control: FCNNodeModel.__init__ (so even
    just being constructed during deserialize runs it once - harmless
    here since input edges don't exist yet at that point, so it fails
    fast on missing inputs), FCNSubWindow.fileLoad -> do_eval_outputs
    (evaluates every node in the scene right after the file opens -
    the actual "starts by opening the file" problem), and undo/redo
    (history.addHistoryRestoredListener -> the same do_eval_outputs).

    A node that shouldn't run on any of those sets
    NodeContent_class = _bootstrap.RunButtonContent and starts its own
    eval_operation with:

        if not self._armed:
            return _bootstrap.not_armed_result(self)
        self._armed = False

    so the real work only happens once its own "Run" button has
    actually been clicked - `_armed` is a plain instance attribute
    (defaults to falsy via getattr, no __init__ needed), set True only
    by the button's own click handler below.
    """

    def initUI(self):
        self.setStyleSheet("background:transparent;")
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 20, 4, 4)
        self.setLayout(layout)
        button = QPushButton("▶ Run")
        button.clicked.connect(self._on_click)
        layout.addWidget(button)

    def _on_click(self):
        self.node._armed = True
        self.node.markDirty()
        self.node.eval()


def not_armed_result(node) -> list:
    """Placeholder eval_operation() return for a not-yet-triggered
    armed-run node: an empty (falsy) list for every output, regardless
    of what that output actually represents - a text Summary, a
    Quaternion, a Parameters dict, ... - so every downstream node's own
    "is this connected" guard (`if not some_input: raise ValueError`)
    catches it safely instead of receiving a placeholder string where a
    specific structured type was expected. The hint goes on the node's
    own tooltip instead of into a socket."""
    try:
        node.grNode.setToolTip("Click ▶ Run to start - not evaluated automatically.")
    except Exception:
        pass
    return [[] for _ in node.outputs]


def resolve_path(path_str) -> Path:
    """Resolve a path typed into a Text node against SANDBOX_ROOT if it
    isn't already absolute.

    Every shape_reco node that takes a plain path string (checkpoint
    dir, input/output STL, workdir) uses this - a relative path (the
    convention used by the example graphs and README, e.g. "checkpoints"
    or "dataset_gen/local_dataset/...") must not depend on FreeCAD's
    actual working directory, which depends on how FreeCAD was launched
    and is not something a saved node graph should have to assume.
    """
    path = Path(path_str)
    return path if path.is_absolute() else SANDBOX_ROOT / path
