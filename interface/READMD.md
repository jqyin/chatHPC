# ChatHPC UI

ChatHPC UI deploys a simple web-based chatbot UI as well as an OpenAI compatible REST API. These are
deployed at https://obsidian.ccs.ornl.gov/chathpc and https://obsidian.ccs.ornl.gov/chathpc/api,
respectively. To access the deployment you'll need to be on the ORNL network and log in using your
OLCF credentials.

## FastChat Framework

ChatHPC UI uses the [FastChat](https://github.com/lm-sys/FastChat) framework, we add following additional features: 

- Support for [FORGE](https://github.com/at-aaims/forge), which are LLMs pre-trained on scientific publications.  
- Support for retrieval augmented generation (RAG). We use our fined-tuned [UAE](https://huggingface.co/WhereIsAI/UAE-Large-V1) model as an embedder to retrieve HPC documents, and then add the documents as context to the original prompt.
- Context logging and performance measurement. We save the context in the logs and get the response time breakdowns for tuning and debugging. 

## System Requirements
ChatHPC UI is designed to be deployed on a [Kubernetes](https://kubernetes.io/) cluster. The
configuration here is specific to Slate, ORNL's OpenShift cluster. Model inference is run on 2 or
more V100 GPUs. To run ChatHPC UI on different hardware or cluster you'll likely need to tweak the
configuration and deployment.

## Overview

ChatHPC UI will launch a controller pod and one or more worker pods. The controller pod runs
FastChat's controller to mange the workers, as well as the web UI and REST API. Each worker pod runs
FastChat's `vllm_worker` to run a model. The worker pods need access to GPUs.

The model weights are loaded from ORNL's Orion filesystem, from the path specified by the
`CHATHPC_MODELS_REMOTE_DIR` environment variable. In ORNL's SLate cluster it was necessary to have
the controller pod sync the model weights from Orion into a shared k8s `PersistentVolumeClaim` as
the GPU nodes don't have direct access to the Orion shared filesystem. If both the controller and
GPU worker pods in your cluster can access some form of shared filesystem, you can remove this step
and just load the models directly from the shared filesystem.

[`deploy.sh`](./deploy.sh) builds a Docker image locally, pushes it to Slate's image registry and then applies the
k8s yaml config in [`chathpc.yaml`](./chathpc.yaml).

## Configuration
ChatHPC UI will accept configuration options via either the command line or via environment
variables prefixed with `CHATHPC_`. See [`settings.py:ChatHPCSettings`](./chathpc/settings.py) for
the available configuration options. [`chathpc.yaml`](./chathpc.yaml) contains the deployment
configuration. You can adjust the  settings via the environment variables specified in the yaml and
adjust your resource limits as needed. Each worker StatefulSet runs a single model so you can run
and compare results from different LLMs. FastChat also supports scaling, so you should be able run
multiple workers for the same model if you have high request throughput.

## Running Directly
You can use [`cli.py`](./cli.py) script to run model inference directly.
