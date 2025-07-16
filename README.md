# Thunder

## Hackathon Submission (Summer 2025)

# This submission is part of the Summer 2025 Weekly Hackathon.

## Summary of Changes

### Integration Test Improvements

- **initial_block_download_task**
  - Replaced hard `sleep()` with active polling:
    - Peer connection
    - Block synchronization progress
  - Added configurable timeouts and intervals.
  - Improved logging for sync diagnostics.

- **check_peer_connection**
  - Reactivated and integrated.
  - Ensures mutual peer visibility.

### Test Stability Improvements

- Reduced excessive waits:  
  - Replaced `sleep(Duration::from_secs(1))` with shorter or conditional waits.
- Added **RPC readiness check**:  
  - Active polling on `getblockcount()`  
  - New error type: `SetupError::RpcTimeout`.

### Config Changes

- `.cargo/config.toml`  
  - Disabled `-Zlinker-features=-lld` workaround (Nightly bug workaround).
- `.gitignore`
  - Added `*.log`, `/cache/`, etc.

## Objective

Improved sync performance and test reliability for Hackathon measurement (`cargo bench`).  
Target: faster and more robust `initial_block_download`.



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
