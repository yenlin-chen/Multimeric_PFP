#!/bin/bash

COV_MODE=0

SIM=0.95
./batch_download.sh -f stats/OUT-temp.pdbs_to_download-sim_$SIM.txt -o /mnt/hdd/yenlin/data/Protein_Data_Bank/pdb.gz -p

SIM=1
./batch_download.sh -f stats/OUT-temp.pdbs_to_download-sim_$SIM.txt -o /mnt/hdd/yenlin/data/Protein_Data_Bank/pdb.gz -p

# ls -lah ../../data/external/Protein_Data_Bank/pdb | wc -l
