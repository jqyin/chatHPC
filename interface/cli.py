"""
Run the ChatHPC model in the CLI (see fastchat cli for usage)

Use on Frontier or other HPC with something like

```
module load git
module load cray-python

git clone https://code.ornl.gov/jhi/chathpc-ui.git
cd chathpc-ui
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

srun -A STF218 --nodes=1 --ntasks-per-node=1 --time=00:30:00 --pty bash -i
python ./cli.py --model-path .../forge-m-instruct-base1/ --num-gpus=1
"""

import sys
from chathpc.utils import run_fastchat_module_interactive

if __name__ == "__main__":
    run_fastchat_module_interactive("fastchat.serve.cli", sys.argv[1:])
