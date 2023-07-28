#!/bin/bash

for a in $( seq 0 9 ); do
    mkdir snapshots${a}
    # normal gamma is 0.85 for 60 but we have 5 x as much data per epoch so 60/5 and 0.85**5
    python3 ../ClassifierNetwork/train.py --traindata classifier_network_dataset/train --testdata classifier_network_dataset/test1 --save-model-directory snapshots${a} --log-csv training${a}.csv --seed $(( 1337+a )) --gamma 0.4437 --epochs 12

done
