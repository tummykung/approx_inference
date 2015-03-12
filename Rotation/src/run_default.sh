#!/bin/bash
./run.py --compile --name default
echo "See state/exects/default/ for results"
# ./run.py --run --name default --generate-data 3 3 3 4 --iterations 20 --inference 2
set -x # set verbose
java -ea -Xmx5g -cp .:lib/fig.jar:lib/stanford-corenlp-3.5.1.jar:classes/default Main\
 -seed 1234567\
 -execPoolDir state/execs/default\
 -log.stdout True\
 -experimentName default\
 -model LinearChainCRF\
 -dataSource ../data/a.txt\
 -numIters 20\
 -inferType 0\
 -sentenceLength 5\
 -fullySupervised True\
 -debugVerbose False\
 -stateVerbose False\
 -generateData True\
 -numSamples 3000
 set +x # unset verbose