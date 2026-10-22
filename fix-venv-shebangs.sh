#!/usr/bin/env bash
# Fix the console script shebangs of a venv that was moved or renamed.
#
#   fix-venv-shebangs.sh <venv-dir>
#
# venv console scripts hardcode the absolute path of the interpreter they
# were installed for, so renaming the directory leaves pip, rocm-sdk and
# friends pointing at a path that no longer exists - or worse, at a
# different environment.
set -euo pipefail

VENV="${1:?usage: fix-venv-shebangs.sh <venv-dir>}"
VENV="$(cd "$VENV" && pwd)"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "fix-venv-shebangs: $VENV/bin/python is missing" >&2
    exit 1
fi

PY="$VENV/bin/python"
INTERP="$(readlink -f "$PY")"

fixed=0
while IFS= read -r script; do
    head -1 "$script" | grep -qE '^#!.*/bin/python' || continue
    if ! grep -q "^#!$PY" "$script"; then
        sed -i "1s|^#!.*|#!$PY|" "$script"
        fixed=$((fixed + 1))
    fi
done < <(find "$VENV/bin" -maxdepth 1 -type f -perm -u+x)

echo "fix-venv-shebangs: $fixed script(s) in $VENV/bin now use $PY"
echo "fix-venv-shebangs: interpreter is $INTERP"
