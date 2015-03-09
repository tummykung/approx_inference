#!/bin/bash
./run.py --compile --name default
echo "See state/exects/default/ for results"
# ./run.py --run --name default --generate-data 3 3 3 4 --iterations 20 --inference 2
# ./run.py --run --name default -seed 1234567 --Main.inferType 0 -debug_verbose -fully_supervised
# --generate_data true --num_samples 2000 --datasource data/a.txt --debug_verbose false
# --state_verbose false --sanity_check true --learning_verbose false
java -Xmx5g -cp .:lib/fig.jar:lib/stanford-corenlp-3.5.1.jar:classes/default Main\
 -seed 1234567\
 -execPoolDir state/execs/default\
 -log.stdout True\
 -experimentName default\
 -model LinearChainCRF\
 -dataSource ../data/a.txt\
 -numIters 20\
 -inferType 0\
 -fully_supervised 0\
 -debug_verbose False\
 -state_verbose False\
 -generate_data True\
 -num_samples 2000