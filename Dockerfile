# Stable Rust version, as of January 2025. 
FROM rust:1.84-slim-bookworm AS builder
WORKDIR /workspace
COPY . .

RUN cargo build --locked --release

# Runtime stage
FROM debian:bookworm-slim

COPY --from=builder /workspace/target/release/thunder_orchard_app /bin/thunder_orchard_app
COPY --from=builder /workspace/target/release/thunder_orchard_app_cli /bin/thunder_orchard_app_cli

# Verify we placed the binary in the right place, 
# and that it's executable.
RUN thunder_orchard_app --help

ENTRYPOINT ["thunder_orchard_app"]

