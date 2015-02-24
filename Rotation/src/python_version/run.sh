#!/bin/bash

xis=`seq 0.1 0.1 1.0`
folder=mod3

echo "python toy.py mod3 $1 $2 $3 $4 > $folder/theta_$1_$2_$3_$4_xi_infinity&"
for xi in ${xis[@]}; do
    echo "python toy.py mod3 $1 $2 $3 $4 $xi > $folder/theta_$1_$2_$3_$4_xi_$xi"
	# python toy.py mod3 $1 $2 $3 $4 $xi > $folder/theta_$1_$2_$3_$4_$xi&
done
