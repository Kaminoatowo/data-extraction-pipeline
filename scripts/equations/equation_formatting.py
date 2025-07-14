import json
import re
from pathlib import Path
from typing import Dict, List, Any
from scripts.utils.logger import setup_logger
from scripts.utils.content_format import (
    generate_fine_tuning,
    generate_pretraining,
    generate_instruction,
)

logger = setup_logger("equation_formatting")


def format_latex_equation(latex_str: str) -> str:
    """Format LaTeX equation for better readability while keeping LaTeX syntax."""
    result = latex_str.strip()
    result = re.sub(r"\\\\+", r"\\", result)  # Replace multiple backslashes with one
    result = re.sub(r"\s+", " ", result)  # Normalize whitespace
    return result


def format_units(unit: str) -> str:
    """Format unit string for readability."""
    if not unit or unit.lower() == "dimensionless":
        return "dimensionless"

    unit = unit.replace("^{-1}", "⁻¹").replace("^{-2}", "⁻²")
    unit = unit.replace("^{2}", "²").replace("^{3}", "³")
    return unit


def generate_pretraining_text(equation_data: Dict[str, Any]) -> str:
    """Generate a natural-language training example describing the equation."""
    equation = format_latex_equation(equation_data["equation"])
    description = equation_data["description"]
    symbols = equation_data.get("symbols", {})
    conditions = equation_data.get("conditions", [])
    source = equation_data.get("source", "Unknown Source")

    text = f"The equation ${equation}$ represents {description.lower()} "

    if symbols:
        text += "In this equation, "
        parts = []
        for symbol, info in symbols.items():
            desc = info["description"]
            unit = format_units(info["unit"])
            if unit == "dimensionless":
                parts.append(f"${symbol}$ is the {desc.lower()}")
            else:
                parts.append(f"${symbol}$ is the {desc.lower()} (in {unit})")

        if len(parts) > 1:
            text += ",\n ".join(parts[:-1]) + f",\n and {parts[-1]}. "
        else:
            text += parts[0] + ". "

    if conditions:
        text += "This equation is applicable under the following conditions: "
        text += ",\n ".join(conditions).lower() + ". "

    if source != "Unknown Source":
        text += f"This relationship is documented in {source}."

    return generate_pretraining(text)


def generate_finetuning_pair(equation_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate a QA pair for fine-tuning."""
    equation = format_latex_equation(equation_data["equation"])
    description = equation_data["description"]
    symbols = equation_data.get("symbols", {})
    conditions = equation_data.get("conditions", [])

    question = f"What does the equation ${equation}$ represent?"
    answer = f"{description} "

    if symbols:
        answer += "The symbols in this equation represent:\n"
        for symbol, info in symbols.items():
            desc = info["description"]
            unit = format_units(info["unit"])
            if unit == "dimensionless":
                answer += f"• ${symbol}$: {desc}\n"
            else:
                answer += f"• ${symbol}$: {desc} (units: {unit})\n"

    if conditions:
        answer += f"\nThis equation is typically used under these conditions: {', '.join(conditions).lower()}."

    return generate_fine_tuning(question, answer)


def generate_conversation_pair(equation_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate a pair in instruction-response format."""
    equation = equation_data.get("equation", "")
    description = equation_data.get("description", "No explanation provided.")

    instruction = (
        f"What does the following equation represent?\n\n{format_latex_equation(equation)}"
        if equation
        else "What does the following equation represent?"
    )
    return generate_instruction(instruction, description)


def process_json_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load and return a list of equation entries from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    return []


def save_jsonl(data: List[Dict], path: Path):
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(data: List[Dict], path: Path):
    """Write a list of dicts to a standard JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_datasets(
    input_dir: Path, output_dir: Path, include_conversations: bool = True
):
    """Main entry point for generating all datasets."""
    output_dir.mkdir(exist_ok=True)
    pretraining_data, finetuning_data, conversation_data = [], [], []

    files = list(input_dir.glob("*.json"))
    if not files:
        logger.warning(f"No JSON files found in {input_dir}")
        return

    logger.info(f"Processing {len(files)} JSON files...")

    for file in files:
        logger.info(f"Reading {file.name}")
        equations = process_json_file(file)
        for entry in equations:
            pretraining_data.append(generate_pretraining_text(entry))
            finetuning_data.append(generate_finetuning_pair(entry))
            if include_conversations:
                conversation_data.append(generate_conversation_pair(entry))

    save_jsonl(pretraining_data, output_dir / "pretraining_dataset.jsonl")
    save_jsonl(finetuning_data, output_dir / "finetuning_dataset.jsonl")

    if include_conversations:
        save_json(conversation_data, output_dir / "conversation_dataset.json")

    logger.info("✅ Dataset generation complete.")
    logger.info(f"Pretraining examples: {len(pretraining_data)}")
    logger.info(f"Fine-tuning pairs: {len(finetuning_data)}")
    if include_conversations:
        logger.info(f"Conversation pairs: {len(conversation_data)}")


def format_equation_conversations(input_dir: Path, output_dir: Path):
    """Utility to format conversation pairs from a single file."""
    logger.info(f"Formatting conversations from: {input_dir}")
    # data = process_json_file(input_dir)
    # if not data:
    #     return

    # conversations = [generate_conversation_pair(entry) for entry in data]
    # output_json.parent.mkdir(parents=True, exist_ok=True)
    # save_json(conversations, output_json)
    # logger.info(f"Saved conversation dataset to {output_json}")

    generate_datasets(input_dir, output_dir, include_conversations=True)
