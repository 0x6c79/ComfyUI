#!/usr/bin/env bash
set -euo pipefail

# 同步上游 ComfyUI 到本地 nix 分支，并推回 fork。
cd "$(git rev-parse --show-toplevel)"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "有未提交的改动，先 commit 或还原：" >&2
  git status --short --untracked-files=no >&2
  exit 1
fi

git switch --quiet nix
git fetch upstream

if ! git rebase upstream/master; then
  echo "rebase 没完成：解决冲突后 git rebase --continue，或 git rebase --abort 放弃。" >&2
  exit 1
fi

git push --force-with-lease origin nix

echo "upstream/master $(git rev-parse --short upstream/master)"
echo "nix             $(git rev-parse --short HEAD) 已推送"
echo "若 flake.nix / flake.lock 有变化，记得 direnv reload"
