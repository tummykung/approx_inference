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
 -fullySupervised False\
 -predictionVerbose True\
 -wordSource ../data/100_words.txt\
 -inferType 0\
 -debugVerbose False\
 -stateVerbose False\
 -generateData False\
 -numSamples 3000\
 -gradientDescentType 2\
 -lambda1 0.0\
 -lambda2 0.0\
 -alpha 1.0
 set +x # unset verbose