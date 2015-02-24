#!/bin/bash
./run.py --compile --name default
echo "See state/exects/default/ for results"
./run.py --run --name default --generate-data 3 3 3 4 --iterations 20 --inference 2
