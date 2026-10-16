#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/angr-binaries"
SAMPLES_COMMIT=7fd826a16b511727148f727150cea1d7ba6985f5

if [ -e "$TARGET_DIR/.git" ]; then
  echo "Playground already exists: $TARGET_DIR"
else
  git clone --depth 1 https://github.com/angr/binaries "$TARGET_DIR"
  echo "Playground binaries cloned to $TARGET_DIR"
fi

# These example fixtures are not part of angr/binaries. Fetch data only from a
# fixed revision without installing angr-utils or changing the corpus revision.
TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/bingraph-samples.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT
git -C "$TEMP_DIR" init --quiet
git -C "$TEMP_DIR" fetch --quiet --depth 1 \
  https://github.com/axt/angr-utils "$SAMPLES_COMMIT"
git -C "$TEMP_DIR" archive FETCH_HEAD examples/samples | tar -xf - -C "$TEMP_DIR"

SAMPLES_DIR="$TARGET_DIR/tests/samples"
# Validate every existing file before copying anything, avoiding partial updates
# on conflicts. Additional user-managed files in this directory are left alone.
for sample in "$TEMP_DIR/examples/samples/"*; do
  destination="$SAMPLES_DIR/$(basename "$sample")"
  if [ -e "$destination" ] || [ -L "$destination" ]; then
    if ! cmp -s "$sample" "$destination"; then
      echo "Sample differs from pinned upstream; refusing to overwrite: $destination" >&2
      exit 1
    fi
  fi
done
mkdir -p "$SAMPLES_DIR"
for sample in "$TEMP_DIR/examples/samples/"*; do
  destination="$SAMPLES_DIR/$(basename "$sample")"
  if [ ! -e "$destination" ]; then
    cp -n "$sample" "$destination"
  fi
done
echo "Supplemental samples verified at $SAMPLES_COMMIT: $SAMPLES_DIR"
