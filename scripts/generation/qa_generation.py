import json
from pathlib import Path
from config.config import openai_client, MODEL_NAME
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts

logger = setup_logger("qa_generation")


def generate_qa_from_text(text: str, qa_prompt: str, debug: bool) -> list:
    """
    Sends a RAG/CPT-style text to GPT to generate question-answer pairs.
    Returns a list of dictionaries: [{"question": ..., "answer": ...}, ...]
    """
    logger.debug("Sending text to GPT for Q&A generation...")
    if debug:
        logger.debug(
            f"Debug mode enabled, skipping Q&A generation for text of length {len(text)}"
        )
        return []
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": qa_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.7,
    )

    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Q&A JSON from model: {e}")
        return []


def generate_qa_pairs(
    input_dir: Path, output_dir: Path, prompts_path: str, debug: bool = False
):
    """
    Generates Q&A pairs from each RAG or CPT text file in input_dir.
    """
    logger.info(f"Generating Q&A pairs from files in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)

    for txt_file in sorted(input_dir.glob("*.txt")):
        with txt_file.open("r", encoding="utf-8") as f:
            text = f.read()

        qa_pairs = generate_qa_from_text(text, prompts["qa_prompt"], debug)

        if qa_pairs:
            output_path = output_dir / f"{txt_file.stem}_qa.json"
            with output_path.open("w", encoding="utf-8") as out_f:
                json.dump(qa_pairs, out_f, indent=2, ensure_ascii=False)
            logger.info(f"Saved Q&A pairs to {output_path}")
        else:
            logger.warning(f"No Q&A generated for {txt_file}")
