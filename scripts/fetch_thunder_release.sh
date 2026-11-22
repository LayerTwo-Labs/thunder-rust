#!/usr/bin/env bash
# Download a released thunder_app binary from GitHub, check it against a
# pinned SHA-256, and install it at DEST. The integration tests use it as an
# old peer, older than MIN_PEER_VERSION.
#
# Usage: fetch_thunder_release.sh VERSION TARGET DEST
#
# Only versions and targets pinned below can be fetched. Pin a new one with
# `shasum -a 256` of the release asset.

set -euo pipefail

REPO='LayerTwo-Labs/thunder-rust'

[ $# -eq 3 ] || { echo "Usage: $(basename "$0") VERSION TARGET DEST" >&2; exit 2; }
VERSION="$1"
TARGET="$2"
DEST="$3"

case "$VERSION/$TARGET" in
    0.18.3/x86_64-unknown-linux-gnu)
        SHA256='68717682e3bb0af604c48e0d55ad60d60b24089338b60c7e22100087521a9e9a' ;;
    0.18.3/x86_64-apple-darwin)
        SHA256='1e8d7d54a87c5067edb1a13cf1bbae974119f4dbf7c8827d67a520d85f7dca05' ;;
    0.18.3/aarch64-apple-darwin)
        SHA256='c565f424c3aae1ecff372685e84e1f1960545c1f9b6d359b8eb0f6300e2bcf7a' ;;
    *)
        echo "no pinned thunder $VERSION build for $TARGET" >&2
        exit 1 ;;
esac

command -v curl >/dev/null || { echo "'curl' is required" >&2; exit 1; }
if command -v sha256sum >/dev/null; then
    sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
    sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

if [ -f "$DEST" ] && [ "$(sha256 "$DEST")" = "$SHA256" ]; then
    echo "thunder $VERSION already at $DEST"
    exit 0
fi

URL="https://github.com/$REPO/releases/download/v$VERSION/thunder-$VERSION-$TARGET"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$URL" -o "$TMP/thunder_app"
ACTUAL="$(sha256 "$TMP/thunder_app")"
if [ "$ACTUAL" != "$SHA256" ]; then
    echo "SHA-256 mismatch for $URL: expected $SHA256, got $ACTUAL" >&2
    exit 1
fi
chmod +x "$TMP/thunder_app"
mkdir -p "$(dirname "$DEST")"
mv "$TMP/thunder_app" "$DEST"
echo "Installed thunder $VERSION at $DEST"
