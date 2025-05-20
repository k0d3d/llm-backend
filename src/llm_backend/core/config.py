import os
from enum import Enum
from typing import List, Union, Optional
from pydantic import AnyHttpUrl, ConfigDict, validator


class AppConfig():
    """
    Config for settings classes that allows for
    combining Setings classes with different env_prefix settings.

    Taken from here:
    https://github.com/pydantic/pydantic/issues/1727#issuecomment-658881926
    """

    model_config = ConfigDict(case_sensitive=True, validate_default=False)

    @classmethod
    def prepare_field(cls, field) -> None:
        if "env_names" in field.field_info.extra:
            return
        return super().prepare_field(field)


class AppEnvironment(str, Enum):
    """
    Enum for app environments.
    """

    LOCAL = "local"
    PREVIEW = "preview"
    PRODUCTION = "production"


class Settings:
    """
    Application settings.
    """

    PROJECT_NAME: str = "llm-backend"
    API_PREFIX: str = "/api"
    LOG_LEVEL: str = "WARNING"
    IS_PULL_REQUEST: bool = False
    RENDER: bool = False

    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @property
    def VERBOSE(self) -> bool:
        """
        Used for setting verbose flag in LlamaIndex modules.
        """
        return self.LOG_LEVEL == "DEBUG" or self.IS_PULL_REQUEST or not self.RENDER

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    @validator("LOG_LEVEL", pre=True)
    def assemble_log_level(cls, v: str) -> str:
        """Preprocesses the log level to ensure its validity."""
        v = v.strip().upper()
        if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("Invalid log level: " + str(v))
        return v

    @property
    def ENVIRONMENT(self) -> AppEnvironment:
        """Returns the app environment."""
        if self.RENDER:
            return AppEnvironment.PRODUCTION
        else:
            return AppEnvironment.LOCAL

    @property
    def UVICORN_WORKER_COUNT(self) -> int:
        if self.ENVIRONMENT == AppEnvironment.LOCAL:
            return 2

        return 6

    class Config(AppConfig):
        model_config = ConfigDict(env_prefix="", validate_default=False)


settings = Settings()
# os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
