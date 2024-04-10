#!/bin/bash
while true
do
	rocm-smi  --showuse --showmemuse --showpower --csv | head -n -1 | tail -n +2 | cut -d, -f2-5
	sleep 1
done 
