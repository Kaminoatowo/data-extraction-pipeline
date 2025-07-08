import yaml
from pathlib import Path


def load_prompts(yaml_path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
