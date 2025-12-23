#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"

if python3 -c 'import matplotlib' >/dev/null 2>&1; then
  echo "OK: matplotlib already available in system python."
  exit 0
fi

if python3 -m venv "${VENV_DIR}" >/dev/null 2>&1; then
  "${VENV_DIR}/bin/pip" install -U pip
  "${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/plot_requirements.txt"
  echo "OK: created ${VENV_DIR}"
  echo "Run: ${VENV_DIR}/bin/python ${SCRIPT_DIR}/summarize_experiments.py --root ${SCRIPT_DIR}/experiments --out ${SCRIPT_DIR}/experiments_summary --per-experiment-plots"
  exit 0
fi

echo "WARN: python venv is not available (ensurepip missing). Falling back to apt install..." >&2
apt-get update
apt-get install -y python3-matplotlib python3-numpy

echo "OK: installed python3-matplotlib (system)."
echo "Run: python3 ${SCRIPT_DIR}/summarize_experiments.py --root ${SCRIPT_DIR}/experiments --out ${SCRIPT_DIR}/experiments_summary --per-experiment-plots"
