#!/bin/bash
# Run one heavy job in one of 3 slots (the user's rule 2026-09-26: at most 3 reconstruction runs at once on the laptop).
#   scripts/run_slot.sh [--exclusive] -- command args...
# --exclusive takes all 3 slots (the frame and the liner, ~5 GB), only when all three are free at once and no small job
# is waiting (small jobs register in $SLOTS/wait.<pid> while they wait): the small jobs go first, the heavy ones never
# hold a slot while waiting. Before starting, waits until >= 4 GB are available.
set -u
SLOTS=${SHAPE_RECO_SLOT_DIR:-$HOME/.cache/shape_reco/slots}; mkdir -p "$SLOTS"
EXCL=0; if [ "${1:-}" = "--exclusive" ]; then EXCL=1; shift; fi
[ "${1:-}" = "--" ] && shift
wait_memory() { while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 4 ]; do sleep 10; done; }
small_waiting() {
  local f
  for f in "$SLOTS"/wait.*; do
    [ -e "$f" ] || continue
    if kill -0 "${f##*.}" 2>/dev/null; then return 0; fi
    rm -f "$f"                                      # a waiter that died without cleaning up
  done
  return 1
}
if [ $EXCL = 1 ]; then
  while true; do
    if ! small_waiting; then
      exec 7>"$SLOTS/slot0" 8>"$SLOTS/slot1" 9>"$SLOTS/slot2"
      if flock -n 7 && flock -n 8 && flock -n 9; then wait_memory; "$@"; rc=$?; exit $rc; fi
      exec 7>&- 8>&- 9>&-                           # closing the files drops whatever was locked
    fi
    sleep 15
  done
fi
touch "$SLOTS/wait.$$"; trap 'rm -f "$SLOTS/wait.$$"' EXIT
while true; do
  for k in 0 1 2; do
    exec 9>"$SLOTS/slot$k"
    if flock -n 9; then rm -f "$SLOTS/wait.$$"; wait_memory; "$@"; rc=$?; exit $rc; fi
    exec 9>&-
  done
  sleep 5
done
