#!/usr/bin/env bash
#
# Switch between the ROCm environments kept next to the repo.
#
#   ./switch-env.sh                 list the environments and mark the active one
#   ./switch-env.sh 7.2             activate the env whose HIP version starts with 7.2
#   ./switch-env.sh .venv-rocm101-backup
#
# The active environment is always $COMFYUI_ROOT/.venv, the parked ones are
# .venv-rocm<hip>-backup directories. Switching renames directories, so the
# console script shebangs of both environments are repaired afterwards.
#
# Environment overrides:
#   COMFYUI_ROOT   project directory that owns the venvs (default: this script's dir)

set -euo pipefail

REPO_ROOT="${COMFYUI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE="$REPO_ROOT/.venv"

fail() {
    echo "switch-env: $*" >&2
    exit 1
}

# Reading the torch dist-info is instant, importing torch is not.
torch_version() {
    local env="$1" dist base
    dist="$(find "$env/lib" -maxdepth 3 -type d -name 'torch-[0-9]*.dist-info' -print -quit 2>/dev/null)" || true
    [[ -n "$dist" ]] || return 1
    base="${dist##*/}"
    base="${base#torch-}"
    echo "${base%.dist-info}"
}

hip_version() {
    local version
    version="$(torch_version "$1")" || return 1
    [[ "$version" == *+rocm* ]] || return 1
    echo "${version##*+rocm}"
}

# 7.2.53211 -> 72, 7.14.0 -> 714, 10.1.0 -> 101. Nightly HIP versions carry
# a build number as the patch component, so anything >= 100 is dropped, and a
# trailing zero patch is left out the same way people say the version.
venv_slug() {
    local version="$1" slug="" part
    local -a parts
    IFS=. read -r -a parts <<< "$version"
    while (( ${#parts[@]} > 1 )) && [[ "${parts[-1]}" == "0" ]]; do
        unset 'parts[-1]'
    done
    for part in "${parts[@]}"; do
        [[ "$part" =~ ^[0-9]+$ ]] || break
        (( part >= 100 )) && break
        slug+="$part"
    done
    echo "$slug"
}

env_dirs() {
    local dir
    for dir in "$REPO_ROOT"/.venv*; do
        [[ -d "$dir" && -x "$dir/bin/python" ]] && printf '%s\n' "$dir"
    done
    return 0
}

list_envs() {
    local env name torch hip
    printf '%-30s %-34s %s\n' "ENV" "TORCH" "HIP"
    while IFS= read -r env; do
        name="${env##*/}"
        torch="$(torch_version "$env" || true)"
        hip="$(hip_version "$env" || true)"
        [[ "$env" == "$ACTIVE" ]] && name="$name *"
        printf '%-30s %-34s %s\n' "$name" "${torch:-?}" "${hip:-?}"
    done < <(env_dirs | sort)
}

# Compare dot separated version prefixes: 7.2 matches 7.2.53211, 10.1 matches 10.1.0.
version_matches() {
    local want="$1" have="$2" i
    local -a want_parts have_parts
    IFS=. read -r -a want_parts <<< "$want"
    IFS=. read -r -a have_parts <<< "$have"
    (( ${#want_parts[@]} <= ${#have_parts[@]} )) || return 1
    for i in "${!want_parts[@]}"; do
        [[ "${want_parts[i]}" == "${have_parts[i]}" ]] || return 1
    done
}

resolve_target() {
    local want="$1" dir name hip torch
    [[ -d "$want" ]] && { (cd "$want" && pwd); return 0; }
    [[ -d "$REPO_ROOT/$want" ]] && { (cd "$REPO_ROOT/$want" && pwd); return 0; }

    local -a matches=()
    while IFS= read -r dir; do
        name="${dir##*/}"
        if [[ "$name" == "$want" ]]; then
            matches+=("$dir")
            continue
        fi
        hip="$(hip_version "$dir" || true)"
        torch="$(torch_version "$dir" || true)"
        if [[ -n "$hip" ]] && version_matches "$want" "$hip"; then
            matches+=("$dir")
        elif [[ -n "$torch" ]] && version_matches "$want" "${torch%%+*}"; then
            matches+=("$dir")
        fi
    done < <(env_dirs)

    (( ${#matches[@]} )) || return 1
    if (( ${#matches[@]} > 1 )); then
        printf '%s\n' "${matches[@]##*/}" >&2
        return 2
    fi
    echo "${matches[0]}"
}

case "${1:-list}" in
    list | --list)
        list_envs
        exit 0
        ;;
    -h | --help)
        sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

(( $# == 1 )) || fail "usage: switch-env.sh [list | <version> | <venv-dir>]"
[[ -x "$ACTIVE/bin/python" ]] || fail "no environment at $ACTIVE"

target="$(resolve_target "$1")" || {
    status=$?
    (( status == 2 )) && fail "more than one environment matches '$1'"
    fail "no environment matches '$1' (run switch-env.sh to list them)"
}

if [[ "$target" == "$ACTIVE" ]]; then
    echo "switch-env: $ACTIVE is already active"
    exit 0
fi

if pgrep -f "$ACTIVE/bin/python" >/dev/null 2>&1; then
    fail "a process is still running from $ACTIVE, stop it before switching"
fi

active_hip="$(hip_version "$ACTIVE" || true)"
slug="$(venv_slug "$active_hip")"
[[ -n "$slug" ]] || slug=unknown
backup="$REPO_ROOT/.venv-rocm${slug}-backup"
n=2
while [[ -e "$backup" ]]; do
    backup="$REPO_ROOT/.venv-rocm${slug}-backup-$n"
    n=$((n + 1))
done

mv "$ACTIVE" "$backup"
if ! mv "$target" "$ACTIVE"; then
    mv "$backup" "$ACTIVE"
    fail "could not move ${target##*/} into place"
fi

"$SCRIPT_DIR/fix-venv-shebangs.sh" "$backup" >/dev/null
"$SCRIPT_DIR/fix-venv-shebangs.sh" "$ACTIVE" >/dev/null

echo "switch-env: active is now $ACTIVE"
echo "  torch $(torch_version "$ACTIVE")"
echo "  parked ${backup##*/}"
