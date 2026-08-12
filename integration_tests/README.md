# Integration tests

## Developing
Integration tests are gated behind the `integration-tests` feature.

To run integration tests, run
```sh
cargo run --example integration_tests
```

## Setup

The tests drive a real enforcer, bitcoind and electrs. The quickest way to get
those in place is

```sh
./scripts/setup_integration_tests.sh
```

which fetches or builds the binaries and writes their paths to
`integrationtests.env` in the repo root. 

```sh
cargo run --example integration_tests
```

passing a test name after `--` to run a single test.

An env file is only a convenience: setting the variables in the environment
works just as well, and variables that are already set take precedence over
`integrationtests.env`. To load a different env file, point
`THUNDER_INTEGRATION_TEST_ENV` at it. An example is provided
[here](/integration_tests/example.env).
