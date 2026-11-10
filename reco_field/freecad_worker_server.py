# Runs inside freecadcmd (reco_field/freecad_worker.py starts it): one JSON job per stdin line
# {"script", "env", "cwd", "tag"}; the script runs with runpy as __main__ under exactly that environment and working
# directory, then "@@DONE <tag> <rc>" is printed (rc 0, the SystemExit code, or 1 on an exception). Open FreeCAD
# documents are closed after every job so one job cannot see another's objects.
import json
import os
import runpy
import sys
import traceback

import FreeCAD

base_path = list(sys.path)
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    job = json.loads(raw)
    rc = 0
    os.environ.clear()
    os.environ.update(job["env"])
    sys.path[:] = base_path
    try:
        os.chdir(job.get("cwd") or os.getcwd())
        sys.argv = [job["script"]]
        runpy.run_path(job["script"], run_name="__main__")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        traceback.print_exc(file=sys.stdout)
        rc = 1
    for name in list(FreeCAD.listDocuments()):
        try:
            FreeCAD.closeDocument(name)
        except Exception:
            pass
    sys.stdout.write("\n@@DONE %s %d\n" % (job["tag"], rc))
    sys.stdout.flush()
