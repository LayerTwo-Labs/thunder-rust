# Stable Rust version, as of January 2025. 
FROM rust:1.84-slim-bookworm AS builder
WORKDIR /workspace
COPY . .

RUN cargo build --locked --release

# Runtime stage
FROM debian:bookworm-slim

COPY --from=builder /workspace/target/release/photon_app /bin/photon_app
COPY --from=builder /workspace/target/release/photon_app_cli /bin/photon_app_cli

# Verify we placed the binary in the right place, 
# and that it's executable.
RUN photon_app --help

ENTRYPOINT ["photon_app"]

