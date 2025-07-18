# Thunder

## Building

Check out the repo with `git clone`, and then

```bash
$ git submodule update --init
$ cargo build
```

## Running

```bash
# Starts the RPC-API server
$ cargo run --bin thunder_app -- --headless

# Runs the CLI, for interacting with the JSON-RPC server
$ cargo run --bin thunder_app_cli

# Runs the user interface. Includes an embedded 
# version of the JSON-RPC server. 
$ cargo run --bin thunder_app -- --headless
```

## Benchmarks

```bash
$ RUSTFLAGS="-C target-cpu=native" cargo bench --package thunder --benches --features "bench"
```

Alternatively, if GNU `bc` and `jq` is installed, one can run `scripts/bench.sh` from the root of this repo.
