#!/usr/bin/env bash
set -euo pipefail
# Explicitly run this on a rented CUDA host, from repository root.
for variant in B0 L3 M1 M2 G2; do
    nextbert train --config "configs/${variant}.json"
done
