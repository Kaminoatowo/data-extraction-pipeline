import json
from pathlib import Path
from scripts.utils.logger import setup_logger

logger = setup_logger("equation_formatting")


def format_equation_conversations(json_file: Path, output_path: Path):
    """
    Converts a JSON list of equations into instruction → response pairs.
    """
    logger.info(f"Formatting conversation data from: {json_file}")
    try:
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        conversations = []
        for item in data.get("equations", []):
            prompt = f"What does the following equation represent?\n\n{item.get('equation', '')}"
            answer = item.get("explanation", "No explanation provided.")
            conversations.append({"instruction": prompt, "response": answer})

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as out:
            json.dump(conversations, out, indent=2, ensure_ascii=False)

        logger.info(f"Saved formatted conversations to {output_path}")
    except Exception as e:
        logger.error(f"Error formatting {json_file}: {e}")
