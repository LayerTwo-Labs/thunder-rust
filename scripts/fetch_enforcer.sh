#!/usr/bin/env bash
# Download the bip300301_enforcer build of COMMIT (a full hash) from
# releases.ecash.com, verify its provenance, and install the binary at DEST.
#
# Usage: fetch_enforcer.sh COMMIT TARGET DEST
#
# Every enforcer master commit is published to nightly/<commit[:12]>/;
# commits not on master have no build.

set -euo pipefail

ECASH_RELEASES="${ECASH_RELEASES:-https://releases.ecash.com}"
PROJECT='bip300301-enforcer'
REPO='LayerTwo-Labs/bip300301_enforcer'
SIGNER_WORKFLOW='ecash-com/releases/.github/workflows/publish.yml'
SOURCE_REF='refs/heads/master'

[ $# -eq 3 ] || { echo "Usage: $(basename "$0") COMMIT TARGET DEST" >&2; exit 2; }
COMMIT="$1"
TARGET="$2"
DEST="$3"
case "$COMMIT" in
    *[!0-9a-f]* | '') echo "invalid commit '$COMMIT'" >&2; exit 2 ;;
esac
[ ${#COMMIT} -eq 40 ] || { echo "commit must be a full 40-character hash, got '$COMMIT'" >&2; exit 2; }

for tool in curl jq unzip gh; do
    command -v "$tool" >/dev/null || { echo "'$tool' is required" >&2; exit 1; }
done

RELEASE_URL="$ECASH_RELEASES/$PROJECT/nightly/${COMMIT:0:12}"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

if ! curl -fsSL "$RELEASE_URL/release.json" -o "$TMP/release.json"; then
    echo "no enforcer build of $COMMIT at $RELEASE_URL/" >&2
    echo "Only commits pushed to $REPO master are published." >&2
    exit 1
fi
# The directory is named by a 12-character prefix; make sure it holds this
# commit and not another one sharing the prefix.
RELEASE_COMMIT="$(jq -er '.commit' "$TMP/release.json")"
if [ "$RELEASE_COMMIT" != "$COMMIT" ]; then
    echo "release.json at $RELEASE_URL/ is for $RELEASE_COMMIT, not $COMMIT" >&2
    exit 1
fi
VERSION="$(jq -er '.version' "$TMP/release.json")"
ZIP_NAME="$(jq -er --arg target "$TARGET" \
    '.files[] | select(.target == $target) | .name' "$TMP/release.json")" || {
    echo "enforcer ${COMMIT:0:12} publishes no zip for target '$TARGET'" >&2
    exit 1
}
case "$ZIP_NAME" in
    *[!A-Za-z0-9._-]* | */*) echo "release.json: bad zip name '$ZIP_NAME'" >&2; exit 1 ;;
esac
EXPECTED_SHA256="$(jq -er --arg name "$ZIP_NAME" \
    '.files[] | select(.name == $name) | .sha256' "$TMP/release.json")"

echo "Downloading bip300301_enforcer $VERSION ($TARGET)..."
curl -fsSL "$RELEASE_URL/provenance.jsonl" -o "$TMP/provenance.jsonl"
curl -# -fL "$RELEASE_URL/$ZIP_NAME" -o "$TMP/$ZIP_NAME"

verify() {
    gh attestation verify "$1" \
        --bundle "$TMP/provenance.jsonl" \
        --repo "$REPO" \
        --signer-workflow "$SIGNER_WORKFLOW" \
        --source-ref "$SOURCE_REF" \
        --source-digest "$COMMIT" \
        --deny-self-hosted-runners
}
sha256() {
    if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1
}

echo "Verifying build provenance..."
verify "$TMP/release.json" || { echo "release.json failed provenance verification" >&2; exit 1; }
ACTUAL_SHA256="$(sha256 "$TMP/$ZIP_NAME")"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "$ZIP_NAME: sha256 $ACTUAL_SHA256 does not match release.json ($EXPECTED_SHA256)" >&2
    exit 1
fi
verify "$TMP/$ZIP_NAME" || { echo "$ZIP_NAME failed provenance verification; not installing it" >&2; exit 1; }

# The zip holds one directory with the bip300301-enforcer binary in it.
mkdir "$TMP/unpacked"
unzip -q "$TMP/$ZIP_NAME" -d "$TMP/unpacked"
UNPACKED_DIRS=("$TMP/unpacked"/*/)
if [ ${#UNPACKED_DIRS[@]} -ne 1 ] || [ ! -f "${UNPACKED_DIRS[0]}bip300301-enforcer" ]; then
    echo "$ZIP_NAME: expected one directory holding bip300301-enforcer, got: $(ls -R "$TMP/unpacked")" >&2
    exit 1
fi

mkdir -p "$(dirname "$DEST")"
mv "${UNPACKED_DIRS[0]}bip300301-enforcer" "$DEST"
chmod +x "$DEST"
echo "Installed bip300301_enforcer $VERSION to $DEST"
