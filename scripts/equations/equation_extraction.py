import json
from pathlib import Path
from config.config import openai_client, MODEL_NAME
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts

logger = setup_logger("equation_extraction")


def extract_equations_from_text(text: str, prompt: str, debug: bool) -> dict:
    """
    Sends the OCR text to the model and returns a JSON-parsable structure with extracted equations.
    """
    logger.debug("Extracting equations with GPT")
    if debug:
        logger.debug(
            f"Debug mode enabled, skipping extraction for text of length {len(text)}"
        )
        return {}
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    try:
        extracted = response.choices[0].message.content
        data = json.loads(extracted)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from model: {e}")
        return {}


def generate_equation_jsons(
    input_txt: Path, output_dir: Path, prompts_path: str, debug: bool = False
):
    """
    Process all OCR .txt files and generate JSON files with extracted equations.
    """
    logger.info(f"Generating equation JSONs from {input_txt}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)
    txt_files = sorted(input_txt.glob("*.txt"))

    for txt_file in txt_files:
        with txt_file.open("r", encoding="utf-8") as f:
            text = f.read()
        result = extract_equations_from_text(
            text, prompts["equation_extraction_prompt"], debug
        )
        if result:
            out_path = output_dir / f"{txt_file.stem}.json"
            with out_path.open("w", encoding="utf-8") as json_file:
                json.dump(result, json_file, indent=2, ensure_ascii=False)
            logger.info(f"Saved equation JSON to {out_path}")
        else:
            logger.warning(f"No equations extracted from {txt_file}")
