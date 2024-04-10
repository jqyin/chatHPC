"""
Helpers for handling app configuration
Based on configutils from Obsidian
"""
from typing import Any, Literal
import sys, argparse
from pathlib import Path
from pydantic import BaseSettings as BaseSettings, Field, constr
from loguru import logger


def cli_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
    """
    A setting source for Pydantic BaseSettings that takes options from the command line.
    Command line options will the same as the field names (or env alias if there is one), converted
    to lowercase kebabcase. E.g. "MY_SPECIAL_FIELD" -> "my-special-field".

    Usage:
    ```python
    class Settings(BaseSettings):
        some_option: datetime

        class Config:
            @classmethod
            def customise_sources(cls, init_settings, env_settings, file_secret_settings):
                return (init_settings, cli_config_settings_source, env_settings, file_secret_settings)
    ```
    ```bash
    my_app.py --some-option='2023-01-01T00:00:00'
    ```
    """
    parser = argparse.ArgumentParser(
        add_help = True,
        exit_on_error = True,
        allow_abbrev = False, # Don't allow abbreviating options to avoid ambiguities
    )
    for (field_name, field) in type(settings).__fields__.items():
        # You can specify multiple aliases, though I haven't found docs for this behavior
        if 'env' in field.field_info.extra: # Use environment variable alias
            env = field.field_info.extra['env']
            names = [env] if isinstance(env, str) else list(env)
        else:
            names = [field_name]

        options = []
        for name in names:
            options.append(f"--{name.lower().replace('_', '-')}")
            if len(name) == 1:
                options.append(f"-{name.lower()}")

        parser.add_argument(*options,
            dest=field_name, # This will make parse_args() output the original field names
            required=False,
            help = field.field_info.description,
        )

    known, unknown = parser.parse_known_args()
    return {k: v for k, v in vars(known).items() if v is not None}


class ChatHPCSettings(BaseSettings):
    role: Literal['controller', 'worker']

    models: str = Field(
        description = """
            Comma separated list of models to run. It will first look for the model in
            models_local_dir, then it will check if its a hugging face repo id.
        """,
    )

    api_port: int = 8080
    api_root_path: str = ""
    ui_port: int = 8081
    ui_root_path: str = ""
    worker_port: int = 21002
    controller_port: int = 21001
    controller_host: str

    # S3 or a mounted gpfs filesystem path
    models_remote_dir: str
    models_local_dir: Path = Path("/vol/data/models")

    num_gpus: int = 1

    log_level: constr(to_upper=True) = Field("INFO", description="One of DEBUG, INFO, WARNING, ERROR")


    def models_list(self) -> list[str]:
        """ Resolves models to list of full paths """
        models = []
        for model in self.models.split(","):
            model = model.strip()
            local_path = (self.models_local_dir / model).resolve()
            if local_path.exists():
                models.append(str(local_path))
            else:
                models.append(model)
        return models


    class Config:
        env_prefix = 'CHATHPC_'

        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (init_settings, cli_config_settings_source, env_settings, file_secret_settings)

    def configure_logger(self, **kwargs):
        """
        Configures the global loguru logger. Should only be called once.
        Takes the same kwargs as loguru Logger.configure, but merges changes with our defaults.
        """
        config = {
            **kwargs,
            "handlers": [
                {"sink": sys.stderr, "level": self.log_level},
                *kwargs.get('handlers', []),
            ],
        }

        # Configure will override the default logger
        logger.configure(**config)
