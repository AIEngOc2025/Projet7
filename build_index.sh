#!/bin/bash
# utility script to fetch data and build the FAISS vector index
# usage: ./build_index.sh

set -euo pipefail

# ensure we're in project root
cd "$(dirname "$0")"

echo "🛠 Building vector index..."
python utilitaires/recuperer_indexer.py

echo "✅ Index building complete."
