# Integration tests

## Developing
Integration tests are gated behind the `integration-tests` feature.

To run integration tests, run
```sh
cargo run --example integration_tests
```

## Setup

The integration tests require at least one environment variable to be set.
Environment variables can also be set via an env file, where the path to the env
file is set via environment variable. An example env file is provided
[here](/integration_tests/example.env). The path to the env file can be provided
by setting the `THUNDER_INTEGRATION_TEST_ENV` variable, eg.

```sh
THUNDER_INTEGRATON_TEST_ENV='integration_tests/example.env'
```

Variables set in an env file have higher precedence than environment variables.
If multiple declarations for the same environment variable exist in an env file,
the last one has highest precedence.
