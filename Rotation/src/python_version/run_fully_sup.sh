#!/bin/bash
folder=fully_supervised_results

echo "python toy.py full $1 $2 $3 $4 > $folder/theta_$1_$2_$3_$4"
python toy.py full $1 $2 $3 $4 > $folder/theta_$1_$2_$3_$4&
