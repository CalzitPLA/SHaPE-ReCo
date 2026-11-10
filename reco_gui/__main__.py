import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))          # the engine imports the generator as `src`


def main():
    parser = argparse.ArgumentParser(description="Step-by-step supervision GUI for the SHaPe ReCo reconstruction")
    parser.add_argument("--stl", help="part to preselect (an STL, or a STEP file in the vector-field mode)")
    parser.add_argument("--random", action="store_true", help="preselect a random STEP file of the downloaded test parts (the Random button)")
    parser.add_argument("--retry-iou", type=float, default=None,
                        help="cascade: ask for another path when a finished run's IoU is below this (0 = never; the panel's 0.95)")
    parser.add_argument("--target-iou", type=float, default=None,
                        help="cascade mode: the level of detail - the IoU the feature phase works up to (the settings' slider; default: all details)")
    parser.add_argument("--subparts", action="store_true",
                        help="cascade mode: sub-parts for the rough model and for pieces without a tool")
    parser.add_argument("--feature-seconds", type=float, default=None, help="cascade mode: the feature phase's time budget")
    parser.add_argument("--max-features", type=int, default=None, help="cascade mode: features applied at most")
    parser.add_argument("--open", default=None, metavar="RUN",
                        help="open a stored run: its directory, or part:latest / part:previous / part:<n> (the toolbar's Open)")
    parser.add_argument("--benchmark", action="store_true",
                        help="open the Benchmark tab: the stored runs of the chosen part, ranked, with their aegir rating")
    parser.add_argument("--benchmark-run", default=None, metavar="RUN",
                        help="with --benchmark: show this stored run at once (its directory or part:latest), with its trees")
    parser.add_argument("--run", action="store_true", help="start at once and run to the end without stopping (live); the pace is the toolbar's")
    parser.add_argument("--method", choices=("cascade", "field"), default=None,
                        help="method to preselect (default: the settings panel's, i.e. the cascade add-cut loop)")
    parser.add_argument("--voxels", type=int, default=None,
                        help="cascade mode: voxels per side of the cube pitch (the settings' 'voxels per side', default 96)")
    parser.add_argument("--start", default=None, help="cascade mode: the starting shape (reco_engine.start_shape.START_TYPES)")
    parser.add_argument("--mode", default=None, choices=("loop", "tree", "search"),
                        help="cascade mode: the rounds - loop, or tree (the decision tree with children, reco_engine/residual_tree.py)")
    parser.add_argument("--max-ops", type=int, default=None, help="cascade mode: operations at most (short runs)")
    parser.add_argument("--features", action=argparse.BooleanOptionalAction, default=None,
                        help="cascade mode: the feature phase after the rough model (the settings' 'Feature phase', on by "
                             "default); --no-features switches it off")
    parser.add_argument("--pace", type=float, default=None, help="seconds to wait after each stage (default: the toolbar's 0.5)")
    parser.add_argument("--follow", default=None, metavar="MIRROR",
                        help="follow a reconstruction running on a server, live (a mirror of scripts/remote_run.py)")
    parser.add_argument("--window", choices=("normal", "maximized", "full"), default="normal",
                        help="the window's size at the start: normal, maximized (the desktop launcher), or full screen "
                             "(F11 toggles it); the docks are fitted to the size")
    parser.add_argument("--verbose", action="store_true",
                        help="the log also goes to the terminal, with a heartbeat line every 10 s while a step computes "
                             "(elapsed, cores in use, open timing section, running child processes), and every "
                             "candidate gets its own log line (the Output panel's 'log every candidate')")
    args = parser.parse_args()
    # VTK's OpenGL widget is reliable on X11; under a Wayland session Qt would otherwise pick the Wayland plugin
    if os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PyQt5 import QtCore, QtGui, QtWidgets
    from reco_gui.app import MainWindow
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(str(ROOT / "packaging" / "shape-reco.png")))   # the logo, packaging/logo/make_logo.py
    app.setDesktopFileName("shape-reco")             # the desktop entry install.sh writes
    method = args.method
    window = MainWindow(stl=args.stl, autorun=False, verbose=args.verbose, autosave=True)
    window.capture_output()                            # what the run prints shows in the Output panel too
    if method is not None:
        combo = window.settings_panel.method
        combo.setCurrentIndex(combo.findData(method))
    if args.stl:
        window.settings_panel.set_input(args.stl)      # again, so the input check sees the chosen method (STEP needs field/cascade)
    if args.random and not args.stl:
        window.settings_panel.pick_random()
    if args.subparts:
        window.settings_panel.c_subparts.setChecked(True)
    if args.max_features is not None:
        window.settings_panel.c_feature_applied.setValue(args.max_features)
    if args.feature_seconds is not None:
        window.settings_panel.c_feature_seconds.setValue(args.feature_seconds)
    if args.target_iou is not None:
        window.settings_panel.set_target_iou(args.target_iou)
    if args.retry_iou is not None:
        window.settings_panel.c_retry_iou.setValue(float(args.retry_iou))
    if args.open and not args.run:
        from reco_engine import run_store
        try:
            window.open_run(run_store.resolve(args.open))
        except ValueError as exc:
            print(f"--open: {exc}")
    if args.benchmark:
        window.show_benchmark()
        if args.benchmark_run:
            from reco_engine import run_store
            try:
                window.show_benchmark_run(run_store.resolve(args.benchmark_run))
            except ValueError as exc:
                print(f"--benchmark-run: {exc}")
    if args.run:                                       # started only now, after the method and the input are set
        window._autorun = True
        from PyQt5 import QtCore
        QtCore.QTimer.singleShot(0, window.start)
    if args.pace is not None:
        window.pace.setValue(args.pace)
    if args.voxels is not None:
        window.settings_panel.c_resolution.setValue(args.voxels)
    if args.start is not None:
        c = window.settings_panel.c_start
        c.setCurrentIndex(max(0, c.findData(args.start)))
    if args.mode is not None:
        c = window.settings_panel.c_mode
        c.setCurrentIndex(max(0, c.findData(args.mode)))
    if args.max_ops is not None:
        window.settings_panel.c_max_ops.setValue(args.max_ops)
    if args.features is not None:
        window.settings_panel.c_features.setChecked(args.features)
    if args.follow:
        window.follow_server_run(args.follow)
    if args.window == "full":
        window.set_full_screen(True)
    elif args.window == "maximized":
        window.showMaximized()
    else:
        window.show()
    if args.window != "normal":                          # once the window manager has given the window its size
        QtCore.QTimer.singleShot(400, window.fit_docks)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
