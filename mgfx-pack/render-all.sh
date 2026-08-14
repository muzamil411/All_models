#!/usr/bin/env bash
# Render every scene with its demo parameters, flattened previews included.
set -euo pipefail
cd "$(dirname "$0")"

for scene in caption card-text card-icons lower-third title-ghost; do
  node render.mjs "$scene" --preview
done

echo
echo "Previews:"
find out -name "*.mp4" -not -path "*/_test/*" | sort
