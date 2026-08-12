#!/usr/bin/env bash
# Set up integration-test dependencies and write integrationtests.env.
# Idempotent. Re-running re-uses cached artifacts.
#
# bitcoind, electrs and the signet miner are the enforcer's dependencies, so
# they are fetched by its own setup script rather than duplicated here. Set
# ENFORCER_ENV to an existing enforcer integrationtests.env to reuse ones
# another checkout already fetched, skipping the downloads.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENFORCER_DIR="$REPO_ROOT/bip300301_enforcer"
ENV_FILE="$REPO_ROOT/integrationtests.env"

if [ ! -f "$ENFORCER_DIR/Cargo.toml" ]; then
    echo "bip300301_enforcer submodule is not checked out." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

# --- Dependencies (bitcoind, electrs, signet miner) ---
if [ -n "${ENFORCER_ENV:-}" ]; then
    if [ ! -f "$ENFORCER_ENV" ]; then
        echo "ENFORCER_ENV is set but '$ENFORCER_ENV' does not exist" >&2
        exit 1
    fi
    echo "Reusing enforcer dependencies from $ENFORCER_ENV"
    DEPS_ENV="$ENFORCER_ENV"
else
    echo "Fetching enforcer dependencies (first run downloads and builds; slow)..."
    "$ENFORCER_DIR/scripts/setup_integration_tests.sh"
    DEPS_ENV="$ENFORCER_DIR/integrationtests.env"
fi

# Read a single-quoted VAR='value' assignment out of an env file.
read_env_var() {
    local file="$1" key="$2" line
    line="$(grep -E "^$key=" "$file" | tail -1 || true)"
    if [ -z "$line" ]; then
        echo "Missing $key in $file" >&2
        exit 1
    fi
    line="${line#"$key"=}"
    # Strip surrounding single quotes, if present.
    line="${line#\'}"
    line="${line%\'}"
    printf '%s' "$line"
}

BITCOIND="$(read_env_var "$DEPS_ENV" BITCOIND)"
BITCOIND_UNPATCHED="$(read_env_var "$DEPS_ENV" BITCOIND_UNPATCHED)"
BITCOIN_CLI="$(read_env_var "$DEPS_ENV" BITCOIN_CLI)"
BITCOIN_UTIL="$(read_env_var "$DEPS_ENV" BITCOIN_UTIL)"
ELECTRS="$(read_env_var "$DEPS_ENV" ELECTRS)"
SIGNET_MINER="$(read_env_var "$DEPS_ENV" SIGNET_MINER)"

# --- Binaries under test ---
# Built from the pinned submodule, so it matches the enforcer library thunder
# compiles against.
echo "Building bip300301_enforcer (from the pinned submodule)..."
cargo build --manifest-path "$ENFORCER_DIR/Cargo.toml" --bin bip300301_enforcer

echo "Building thunder_app..."
cargo build --manifest-path "$REPO_ROOT/Cargo.toml" --bin thunder_app

# --- Env file ---
# Every path is absolute, so the tests do not care what the working directory
# is. The binaries under test are per-checkout. The dependencies may be shared.
cat > "$ENV_FILE" <<EOF
BIP300301_ENFORCER='$ENFORCER_DIR/target/debug/bip300301_enforcer'
BITCOIND='$BITCOIND'
BITCOIND_UNPATCHED='$BITCOIND_UNPATCHED'
BITCOIN_CLI='$BITCOIN_CLI'
BITCOIN_UTIL='$BITCOIN_UTIL'
ELECTRS='$ELECTRS'
SIGNET_MINER='$SIGNET_MINER'
THUNDER_APP='$REPO_ROOT/target/debug/thunder_app'
EOF

echo "Wrote $ENV_FILE"
echo
echo "Run integration tests with:"
echo "  cargo run --example integration_tests [-- <test-name>]"
