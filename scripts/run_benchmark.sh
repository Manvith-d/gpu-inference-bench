#!/usr/bin/env bash
set -e
python benchmark.py --backend dummy --users 200 --concurrency 32 --batch-size 8 --rate 60 --arrival poisson --max-new-tokens 32
echo "Results written to results/summary.csv"
