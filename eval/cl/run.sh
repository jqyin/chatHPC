#!/bin/bash

lm_eval --model hf --model_args pretrained=forge-m2-olcfdoc-200s-olcfdocv4/ --tasks arc_easy,arc_challenge,sciq,piqa,openbookqa  --device cuda:0 --batch_size 32
