# chatHPC: Empowering HPC Users with Large Language Models 
This repository contains the implementation of chatHPC pipeline, including the end-to-end velopment and deployment cycle of LLM applications on HPC. 

<img src="./chathpc.png" width="600">

## Data sources and preprocessing
HPC documentations and OLCF help tickets (contains private and sentitive information)
- OLCF: https://github.com/olcf/olcf-user-docs
- ALCF: https://github.com/argonne-lcf/user-guides
- NERSC: https://gitlab.com/NERSC/nersc.gitlab.io

The pre-processing scripts include for both [documents](./preprocss/process_olcf_doc.py) and [tickets](./preprocss/process_jira_tickets.py). 

## Continual pre-training 
The pre-training on HPC documents uses [FORGE](https://github.com/at-aaims/forge), and a sample input data is [provided](./data/train/model/olcf-user-doc-9-23.jsonl). The model configuration remains the same as the [Forge-13B](https://github.com/at-aaims/forge/blob/main/train/configs/forge-m.yml).

## SI^2 instruction set generation

## Distributed fine-tuning

## RAG

## User interface 

## Evaluation 

## Results 
The plots of the results are generated using this [script](./plot.ipynb), and the corresponding raw job logs can be [downloaded](https://www.dropbox.com/scl/fo/xscwgb8o1d2c47kjwuap1/ABpnF0u4hE8i7JMMLy-7AgI?rlkey=p9fnhfyoqu8o7vh1jbgn6tnms&dl=0)
