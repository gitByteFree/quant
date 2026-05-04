"""YAML配置加载器，支持环境变量替换."""

import os
import re
from pathlib import Path

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env_vars(value: str) -> str:
    """替换字符串中的 ${VAR:default} 模式."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default if default is not None else "")

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _resolve_recursive(obj: object) -> object:
    """递归解析配置对象中的环境变量."""
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _resolve_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_recursive(v) for v in obj]
    return obj


def load_config(path: str | Path) -> dict:
    """加载YAML配置文件并解析环境变量.

    Args:
        path: 配置文件路径

    Returns:
        解析后的配置字典
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _resolve_recursive(config)
