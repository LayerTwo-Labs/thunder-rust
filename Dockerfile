# Stable Rust version, as of January 2025.
FROM rust:1.84-slim-bookworm AS builder
WORKDIR /workspace
COPY . .

RUN cargo build --locked --release

# Runtime stage
FROM debian:bookworm-slim

# Run as a non-root user so a compromised container cannot write host-mapped
# paths as root, and so wallet / cookie files created at runtime are not owned
# by uid 0.
RUN groupadd --system --gid 1000 thunder \
    && useradd --system --uid 1000 --gid thunder --home-dir /var/lib/thunder \
       --shell /usr/sbin/nologin thunder \
    && mkdir -p /var/lib/thunder \
    && chown -R thunder:thunder /var/lib/thunder

COPY --from=builder /workspace/target/release/thunder_app /bin/thunder_app
COPY --from=builder /workspace/target/release/thunder_app_cli /bin/thunder_app_cli

# Verify we placed the binary in the right place,
# and that it's executable.
RUN thunder_app --help

USER thunder
WORKDIR /var/lib/thunder
ENV HOME=/var/lib/thunder

ENTRYPOINT ["thunder_app"]
