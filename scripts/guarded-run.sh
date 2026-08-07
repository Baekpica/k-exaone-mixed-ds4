#!/usr/bin/env bash
# Run a model load/serve under a hard memory ceiling and make it the OOM
# killer's first choice.
#
# Why this exists: GB10 is a unified-memory SoC, so an inference process that
# over-allocates competes directly with the shell, the editor and the tmux
# server for the same 121.6 GiB. The first time that happened here it took the
# tmux window with it.
#
#   scripts/guarded-run.sh [-m MAX_GIB] [-h HIGH_GIB] [-l LOGFILE] -- <cmd>...
#
# Defaults to 112 GiB, which leaves ~9.6 GiB for the OS and the session.
#
# MemoryHigh is the part that matters on this box, and it is not just a safety
# net. Reading an 86 GiB artifact fills the page cache, and DGX Spark does not
# reclaim it promptly under global pressure -- the operator here keeps a
# `drop_caches` alias on hand precisely because of that. A cgroup v2 high
# watermark restores the missing behaviour locally: when the scope approaches
# it, the kernel reclaims *that scope's* page cache instead of letting the
# whole system walk into the OOM killer. MemoryMax above it is the hard stop.
#
# The raised oom_score_adj is the backstop for what cgroup accounting may not
# cover: whether the NVIDIA driver's unified-memory allocations are charged to
# the scope is not guaranteed. If the global OOM killer does fire, this makes
# the loader the victim rather than the session. Both are applied.
set -uo pipefail

LIMIT_GIB=112
HIGH_GIB=""
LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m) LIMIT_GIB="$2"; shift 2 ;;
    -h) HIGH_GIB="$2"; shift 2 ;;
    -l) LOG="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "usage: $0 [-m MAX_GIB] [-h HIGH_GIB] [-l LOGFILE] -- <cmd>..." >&2; exit 2 ;;
  esac
done
[[ $# -gt 0 ]] || { echo "$0: no command given" >&2; exit 2; }
# Start reclaiming before the hard stop, so the scope sheds page cache rather
# than hitting MemoryMax and dying. A fixed 6 GiB gap goes negative on the
# small ceilings used for testing, so take the larger reclaim headroom of
# 6 GiB and 10% of the ceiling.
if [[ -z "$HIGH_GIB" ]]; then
  HIGH_GIB=$(( LIMIT_GIB > 12 ? LIMIT_GIB - 6 : (LIMIT_GIB * 9 + 9) / 10 ))
fi

mem_line() {
  awk -v tag="$1" '/MemTotal/{t=$2}/MemFree/{f=$2}/MemAvailable/{a=$2}/^Cached/{c=$2}
    END{printf "%s total=%.1f free=%.1f avail=%.1f cached=%.1f GiB\n",
        tag, t/2^20, f/2^20, a/2^20, c/2^20}' /proc/meminfo
}

mem_line "guarded-run: before"
echo "guarded-run: reclaim above ${HIGH_GIB} GiB, hard stop ${LIMIT_GIB} GiB, oom_score_adj 800"

# The scope name has to be unique per run or systemd refuses to start it.
SCOPE="ds4-guarded-$$-$(date +%s)"
run() {
  # Raise our own oom badness before exec'ing the payload, so the payload
  # inherits it. Raising is always permitted; lowering would need
  # CAP_SYS_RESOURCE, which is exactly why the protection is "make the loader
  # the victim" rather than "make the session immune".
  echo 800 > /proc/self/oom_score_adj 2>/dev/null || true
  exec "$@"
}
export -f run

if [[ -n "$LOG" ]]; then
  systemd-run --user --scope --quiet --unit="$SCOPE" \
      -p "MemoryHigh=${HIGH_GIB}G" -p "MemoryMax=${LIMIT_GIB}G" -p "MemorySwapMax=0" \
      bash -c 'run "$@"' _ "$@" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
else
  systemd-run --user --scope --quiet --unit="$SCOPE" \
      -p "MemoryHigh=${HIGH_GIB}G" -p "MemoryMax=${LIMIT_GIB}G" -p "MemorySwapMax=0" \
      bash -c 'run "$@"' _ "$@"
  rc=$?
fi

mem_line "guarded-run: after"
if [[ $rc -ne 0 ]]; then
  echo "guarded-run: exit $rc" >&2
  # 137 is SIGKILL, which under a MemoryMax scope means the ceiling was hit --
  # report that plainly instead of leaving a bare signal number.
  [[ $rc -eq 137 ]] && echo "guarded-run: killed at the ${LIMIT_GIB} GiB ceiling" >&2
fi
exit $rc
