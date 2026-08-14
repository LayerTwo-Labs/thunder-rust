default:
    @just --list

fmt:
    cargo fmt

build:
    cargo build

clippy:
    cargo clippy --all-targets --all-features

# Run integration tests. Pass a test name to run a single test:
# `just test-it deposit_withdraw_roundtrip`
test-it *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f integrationtests.env ] && [ -z "${THUNDER_INTEGRATION_TEST_ENV:-}" ]; then
        echo "No integrationtests.env found. Run ./scripts/setup_integration_tests.sh first," >&2
        echo "or point THUNDER_INTEGRATION_TEST_ENV at an existing env file." >&2
        exit 1
    fi
    cargo run --example integration_tests -- {{ args }}
