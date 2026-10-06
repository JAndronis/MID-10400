#!/usr/bin/env bash
# The proposal-shared environment: one interpreter, one deploy clone tracking
# `main`, one Jupyter kernel, all readable by every proposal member and by
# DAMNIT's service account.
#
# Why: uv installs its Python into ~/.local/share/uv/python by default, and a
# venv's bin/python is only a symlink to it. Home directories on Maxwell are
# private, so for anyone but the builder that link dangles, `activate` falls
# through to /usr/bin/python without a word, and DAMNIT jobs cannot start.
#
#   build          interpreter + deploy clone + environment + kernelspec (idempotent)
#   update         move the deploy clone to the source clone's `main`; re-sync the
#                  environment when pyproject.toml or uv.lock changed. The git
#                  hooks run this after every commit, merge and rewrite on `main`
#   install-hooks  install those hooks into the source clone
#   link-kernel    make the shared kernel visible to *your* Jupyter (once per user)
#   venv           sync the .venv of the clone you are in against the shared
#                  interpreter, dev group included (for development clones)

set -euo pipefail
# A hook runs with GIT_DIR & co. pointing at the repository that fired it.
unset $(git rev-parse --local-env-vars) VIRTUAL_ENV
umask 022

PROPOSAL=/gpfs/exfel/exp/MID/202601/p010400
SOFTWARE=$PROPOSAL/usr/Software
SOURCE=$PROPOSAL/usr/Shared/IA/MID-10400
DEPLOY=$SOFTWARE/MID-10400
PYTHON_VERSION=3.12.11
export UV_PYTHON_INSTALL_DIR=$SOFTWARE/uv-python
PYTHON=$UV_PYTHON_INSTALL_DIR/cpython-$PYTHON_VERSION-linux-x86_64-gnu/bin/python3.12
KERNEL_PREFIX=$SOFTWARE/jupyter
KERNEL_NAME=mid-10400
# The uv cache sits in the builder's home directory; never hard-link into it.
export UV_LINK_MODE=copy
# No dev group: zlib_into switches EXtra-data to its slower threaded path.
DEPLOY_SYNC=(--locked --no-dev --extra euxfel --extra plotting)

UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}

say() { echo "shared_env: $*" >&2; }
die() {
    say "ERROR: $*"
    exit 1
}

sync_deploy() { (cd "$DEPLOY" && "$UV" sync "${DEPLOY_SYNC[@]}" --python "$PYTHON"); }

cmd_build() {
    "$UV" python install "$PYTHON_VERSION"
    [ -x "$PYTHON" ] || die "expected interpreter missing: $PYTHON"
    [ -d "$DEPLOY/.git" ] || git clone --branch main "$SOURCE" "$DEPLOY"
    sync_deploy
    "$DEPLOY/.venv/bin/python" -m ipykernel install --prefix "$KERNEL_PREFIX" \
        --name "$KERNEL_NAME" --display-name "MID-10400" \
        --env VIRTUAL_ENV "$DEPLOY/.venv"
    say "built at $(git -C "$DEPLOY" rev-parse --short HEAD)"
}

cmd_update() {
    [ -d "$DEPLOY/.git" ] || die "no deploy clone at $DEPLOY; run '$0 build'"
    cd "$DEPLOY"
    exec 9>.git/shared_env.lock
    flock 9
    [ -z "$(git status --porcelain --untracked-files=no)" ] ||
        die "$DEPLOY has local changes; it only mirrors $SOURCE main"
    local old new
    old=$(git rev-parse HEAD)
    git fetch -q origin main
    new=$(git rev-parse FETCH_HEAD)
    [ "$old" != "$new" ] || return 0
    git reset -q --hard "$new"
    if ! git diff --quiet "$old" "$new" -- pyproject.toml uv.lock ||
        [ ! -x .venv/bin/python ]; then
        if ! sync_deploy; then
            git reset -q --hard "$old"
            sync_deploy || true
            die "uv sync failed at ${new:0:7}; shared environment left at ${old:0:7}"
        fi
    fi
    say "shared environment ${old:0:7} -> ${new:0:7} (restart open kernels)"
}

cmd_install_hooks() {
    local hook
    for hook in post-commit post-merge post-rewrite; do
        cat >"$SOURCE/.git/hooks/$hook" <<'EOF'
#!/bin/sh
# Installed by scripts/shared_env.sh install-hooks.
[ "$(git symbolic-ref --short -q HEAD)" = main ] || exit 0
exec "$(git rev-parse --show-toplevel)/scripts/shared_env.sh" update
EOF
        chmod +x "$SOURCE/.git/hooks/$hook"
    done
    say "hooks installed in $SOURCE/.git/hooks"
}

cmd_link_kernel() {
    local dest=${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/$KERNEL_NAME
    if [ -e "$dest" ] && [ ! -L "$dest" ]; then
        die "$dest already exists; remove it and run again"
    fi
    mkdir -p "$(dirname "$dest")"
    ln -sfn "$KERNEL_PREFIX/share/jupyter/kernels/$KERNEL_NAME" "$dest"
    say "kernel '$KERNEL_NAME' linked at $dest"
}

cmd_venv() {
    cd "$(git rev-parse --show-toplevel)"
    "$UV" sync --locked --extra euxfel --extra plotting --python "$PYTHON"
}

case "${1:-}" in
build) cmd_build ;;
update) cmd_update ;;
install-hooks) cmd_install_hooks ;;
link-kernel) cmd_link_kernel ;;
venv) cmd_venv ;;
*)
    sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2
    exit 2
    ;;
esac
