#!/bin/sh
# One-time setup: a host-side mirror of SWE-CI's Dockerfile.opencode
# (SWE-CI/src/swe_ci/benchmark/agents/Dockerfile.opencode), since there is no
# container on this cluster. Installs a pinned node + opencode-ai under
# FORESIGHT_ROOT/agent and writes a wrapper that execs it with an isolated
# environment. Safe to re-run.
#
# Usage:
#   deploy/tau-slurm/setup/install_opencode.sh [--root DIR]
#
# Run this on the login node -- it needs outbound internet for the node
# tarball and npm.

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "$SCRIPT_DIR/../lib/common.sh"

foresight_resolve_root "$@"

NODE_VERSION="${NODE_VERSION:-22.18.0}"   # pinned to match SWE-CI's Dockerfile.opencode
AGENT_NPM_PKG="${AGENT_NPM_PKG:-opencode-ai}"
AGENT_BIN="${AGENT_BIN:-opencode}"
AGENT_HOME="$(foresight_agent_dir)"

mkdir -p "$AGENT_HOME/node" "$AGENT_HOME/npm-global" "$AGENT_HOME/npm-cache" \
         "$AGENT_HOME/home" "$AGENT_HOME/cache" "$AGENT_HOME/bin"

arch="$(uname -m)"
case "$arch" in
    x86_64) node_arch="x64" ;;
    aarch64|arm64) node_arch="arm64" ;;
    *)
        echo "foresight: unsupported arch: $arch" >&2
        exit 1
        ;;
esac

if [ ! -x "$AGENT_HOME/node/bin/node" ] \
   || [ "$("$AGENT_HOME/node/bin/node" --version 2>/dev/null)" != "v$NODE_VERSION" ]; then
    echo "foresight: fetching node v$NODE_VERSION ($node_arch) ..." >&2
    tmp_tarball="$(mktemp)"
    curl -fsSL \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" \
        -o "$tmp_tarball"
    rm -rf "$AGENT_HOME/node"
    mkdir -p "$AGENT_HOME/node"
    tar -xJf "$tmp_tarball" -C "$AGENT_HOME/node" --strip-components=1
    rm -f "$tmp_tarball"
fi

echo "foresight: installing $AGENT_NPM_PKG ..." >&2
NPM_CONFIG_PREFIX="$AGENT_HOME/npm-global" \
npm_config_cache="$AGENT_HOME/npm-cache" \
PATH="$AGENT_HOME/node/bin:$PATH" \
"$AGENT_HOME/node/bin/npm" install -g "$AGENT_NPM_PKG"

test -x "$AGENT_HOME/npm-global/bin/$AGENT_BIN"

wrapper="$AGENT_HOME/bin/$AGENT_BIN"
cat > "$wrapper" <<EOF
#!/bin/sh
set -eu
BASE="$AGENT_HOME"
export NODE_NO_WARNINGS=1
# SWE-CI's container wrapper hardcodes HOME=\$BASE/home. This one accepts an
# override because aux and target need DIFFERENT homes on Slurm: opencode
# keeps its session SQLite (and, for SWE-CI, its token accounting) under
# \$HOME/.local/share/opencode, and there is no container here to isolate
# them the way "docker exec -e HOME=..." does.
export HOME="\${OPENCODE_HOME:-\$BASE/home}"
export XDG_CACHE_HOME="\$BASE/cache"
export NPM_CONFIG_PREFIX="\$BASE/npm-global"
export npm_config_cache="\$BASE/npm-cache"
export PATH="\$BASE/node/bin:\$BASE/npm-global/bin:\$PATH"
mkdir -p "\$HOME"
exec "\$BASE/npm-global/bin/$AGENT_BIN" "\$@"
EOF
chmod +x "$wrapper"

echo "foresight: done. opencode wrapper at $wrapper" >&2
echo "foresight: give aux and target distinct homes, e.g.:" >&2
echo "  OPENCODE_HOME=\$RUN/homes/target $wrapper run --model custom/target-model \"...\"" >&2
