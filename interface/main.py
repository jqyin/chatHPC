"""
Launch the ChatHPC model
"""
import sys, time, os, socket
import requests
from pathlib import Path
from chathpc.settings import ChatHPCSettings
from chathpc.utils import (
    run_fastchat_module, wait_for_first, test_message, rsync,
)
from loguru import logger

settings = ChatHPCSettings()
settings.configure_logger()

# This app is using the FastChat framework to deploy the model. FastChat handles the tricky parts
# of running, scaling, and deploying the model. It gives us a fully OpenAI compatible REST API 
# and a simple chat UI. It also supports scaling the deployment, which we may need to do later.
# However, it has a few rough edges that we needed to work around.

# The biggest complication is adding our own model configuration. FastChat's workflow for adding
# models intends for you to fork or PR FastChat (see https://github.com/lm-sys/FastChat/blob/main/docs/model_support.md)
# So you can change the configuration for the model in the fork of FastChat in the submodule.

# See also: FastChat README and https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md


def launch_controller():
    model_names = [m.split("/")[-1] for m in settings.models_list()]
    controller_addr = f"http://localhost:{settings.controller_port}"

    # We can't properly disable FastChat logging to file when running the UI, so clear out the log
    # files on bootup so they don't build up too much.
    log_dir = Path(os.environ['LOGDIR'])
    for file in log_dir.glob("*.log"):
        file.unlink()

    # Download the models (if necessary)
    rsync(settings.models_remote_dir, settings.models_local_dir)

    # Launch the controller server.
    controller = run_fastchat_module("fastchat.serve.controller", [
        "--host", "0.0.0.0",
        "--port", str(settings.controller_port)
    ])

    # Hack to wait until the model_worker is up and running
    # TODO: This assumes we scaled down the workers before redploying a new version of model
    loaded_models = []
    while set(loaded_models) != set(model_names):
        if controller.poll(): # Something died
            sys.exit()
        logger.info(f"Waiting for models to load... (loaded: {loaded_models})")
        time.sleep(5)
        loaded_models = test_message("dummy_model", message="Are you there?").loaded_models

    logger.info(f"Loaded models: {loaded_models}")
    
    openai_api = run_fastchat_module("fastchat.serve.openai_api_server", [
        "--host", "0.0.0.0",
        "--port", str(settings.api_port),
        "--root-path", settings.api_root_path,
        "--controller-address", controller_addr,
    ])

    web_ui = run_fastchat_module("fastchat.serve.gradio_web_server", [
        "--host", "0.0.0.0",
        "--port", str(settings.ui_port),
        "--root-path", settings.ui_root_path,
        "--controller-url", controller_addr,
    ])

    # Block until one of the processes dies for some reason
    wait_for_first(controller, openai_api, web_ui)



def launch_worker():
    # Wait for the controller to be launched before starting.
    # This means we know that the volume contains the model files.
    controller_addr = f"http://{settings.controller_host}:{settings.controller_port}"
    model = settings.models_list()[0] # Only one model allowed for each worker
    model_name = model.split("/")[-1]
    worker_addr = f"http://{socket.getfqdn()}:{settings.worker_port}"

    logger.info(f"worker: {worker_addr} controller: {controller_addr}")

    controller_up = False
    while not controller_up:
        logger.info("Waiting for controller...")
        time.sleep(1)
        try:
            response = requests.get(f"{controller_addr}/test_connection", timeout = 5)
            controller_up = response.ok
        except Exception as e:
            controller_up = False
            logger.info(f"{e}")
    logger.info("Controller up!")

    worker_args = [
        "--host", "0.0.0.0",
        "--port", str(settings.worker_port),
        "--worker-address", worker_addr,
        "--controller-address", controller_addr,
        "--model-names", model_name,
        "--model-path", model,
        "--limit-worker-concurrency", 1,
        "--num-gpus", str(settings.num_gpus),
    ]

    model_worker = run_fastchat_module("fastchat.serve.vllm_worker", [
        *worker_args,
        # "--swap-space=8"
        # "--dtype=half",
    ])

    # model_worker = run_fastchat_module("fastchat.serve.model_worker", [
    #     *worker_args,
    #     # "--load-8bit"
    # ])

    # Block until one of the processes dies for some reason
    wait_for_first(model_worker)


def main():
    if settings.role == "controller":
        launch_controller()
    else: # worker
        launch_worker()


if __name__ == "__main__":
    main()
