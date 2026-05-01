import os
from typing import Dict, Any, List

import yaml
import argparse


def _get_pipeline_path() -> str:
    """Парсим только --pipeline, остальные аргументы игнорируем."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pipeline", default=os.environ.get("PIPELINE"))
    args, _ = parser.parse_known_args()

    if not args.pipeline:
        raise ValueError("--pipeline is required (or set env PIPELINE)")
    return args.pipeline


class ConfigManager:
    __instance = None
    __initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(self):
        if ConfigManager.__initialized:
            return

        pipeline_path = _get_pipeline_path()
        self._validate(pipeline_path)

        with open(pipeline_path) as f:
            self.config = yaml.safe_load(f)
            if not self.config.get('pipeline'):
                raise ValueError("Config must start from 'pipeline'")

        for service in self.config["pipeline"]:
            name = service.get("service_name")
            if not name:
                raise ValueError("Each service must have 'service_name'")
            self._validate_required_params(name)

        ConfigManager.__initialized = True

    @staticmethod
    def _validate(path: str):
        if not path.endswith((".yml", ".yaml")):
            raise ValueError("Config must be .yml or .yaml")
        if not os.path.isfile(path):
            raise ValueError(f"File not found: {path}")

    def _validate_required_params(self, name: str):
        service_settings = self.config_service_name(name)
        if service_settings is None:
            raise ValueError(f"Service '{name}' not found in config")

        service_type = service_settings.get("type")

        required_by_type = {
            "reader":      ["out_channel"],
            "inference":   ["in_channel", "out_channel"],
            "registrator": ["in_channel", "out_channel"],
            "sender":      ["in_channel"],
        }

        required_fields = required_by_type.get(service_type, [])

        for field in required_fields:
            if service_settings.get(field) is None:
                raise ValueError(f"Config {service_type} '{name}' must have '{field}'")

    def config_service_type(self, service_type: str) -> List[Dict[str, Any]]:
        return [
            service for service in self.config["pipeline"]
            if service["type"] == service_type
        ]

    def config_service_name(self, name: str) -> Dict[str, Any] | None:
        return next((service for service in self.config["pipeline"] if service["service_name"] == name), None)


if __name__ == "__main__":
    pipeline_config = ConfigManager()

    print(pipeline_config.config_service_type('reader'))
