"""Standalone launcher for the dataset-generation window.

    python3 -m reco_gui.dataset_gen_main
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))          # the generator imports itself as `src`


def main():
    if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PyQt5 import QtGui, QtWidgets
    from reco_gui.dataset_gen_window import DatasetGenWindow
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(str(ROOT / "packaging" / "shape-reco.png")))   # the logo, packaging/logo/make_logo.py
    app.setDesktopFileName("shape-reco")             # the desktop entry install.sh writes
    window = DatasetGenWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
