import json
from pathlib import Path
import re
from config.config import openai_client, MODEL_NAME
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts
from scripts.utils.content_format import generate_fine_tuning
from scripts.utils.batch_processor import process_batches, process_batch_with_validation

logger = setup_logger("qa_generation")

# Batch processing configuration
BATCH_SIZE = 2  # Process multiple files at once


def load_text_files(input_dir: Path):
    """
    Load all text files and return list of (file_path, content) tuples.

    Args:
        input_dir: Directory containing text files

    Returns:
        List of (file_path, content) tuples
    """
    txt_files = sorted(input_dir.glob("*.txt"))
    file_data = []

    for txt_file in txt_files:
        try:
            with txt_file.open("r", encoding="utf-8") as f:
                content = f.read()
            file_data.append((txt_file, content))
            logger.debug(
                f"Loaded text file: {txt_file.name} ({len(content)} characters)"
            )
        except Exception as e:
            logger.error(f"Failed to read {txt_file}: {e}")

    return file_data


def process_qa_batch(batch, batch_idx, qa_prompt, debug=False):
    """
    Process a batch of text files for Q&A generation.

    Args:
        batch: List of (file_path, file_content) tuples
        batch_idx: Index of the current batch
        qa_prompt: The Q&A generation prompt
        debug: Whether debug mode is enabled

    Returns:
        List of generated Q&A pairs from all files in the batch
    """
    batch_qa_pairs = []

    for file_path, text_content in batch:
        logger.debug(f"Generating Q&A for file: {file_path.name}")

        qa_pairs = generate_qa_from_text(text_content, qa_prompt, debug)

        if qa_pairs:
            batch_qa_pairs.extend(qa_pairs)
            logger.info(f"Extracted {len(qa_pairs)} Q&A pairs from {file_path}")
        else:
            logger.warning(f"No Q&A generated for {file_path}")

    return batch_qa_pairs


def extract_qa_pairs(text):
    pairs = []
    qa_chunks = re.findall(
        r"(Q\d?:\s*.+?\nA\d?:\s*.+?)(?=\nQ\d?:|\Z)", text.strip(), re.DOTALL
    )
    for chunk in qa_chunks:
        q_match = re.search(r"Q\d?:\s*(.+)", chunk)
        a_match = re.search(r"A\d?:\s*(.+)", chunk)
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            pairs.append(generate_fine_tuning(question, answer))
    return pairs


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
        content = response.choices[0].message.content.strip()
        qa_pairs = extract_qa_pairs(content)
        return qa_pairs
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Q&A JSON from model: {e}")
        return []


def validate_qa_result(result):
    """
    Validate the Q&A generation result.

    Args:
        result: List of Q&A pairs generated from a batch

    Returns:
        bool: True if the result is valid, False otherwise
    """
    if not result:
        return False

    # Check if at least one Q&A pair was generated
    if len(result) == 0:
        return False

    # Validate that each Q&A pair has the required structure
    for qa_pair in result:
        if not isinstance(qa_pair, dict):
            return False
        # Add more specific validation based on your Q&A structure
        # For example, check if it has 'messages' key for fine-tuning format
        if "messages" not in qa_pair:
            return False

    return True


def generate_qa_pairs(
    input_dir: Path, output_dir: Path, prompts_path: str, debug: bool = False
):
    """
    Generates Q&A pairs from each RAG or CPT text file in input_dir using batch processing.
    """
    logger.info(f"Generating Q&A pairs from files in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)
    qa_prompt = prompts["qa_prompt"]

    # Load all text files
    file_data = load_text_files(input_dir)

    if not file_data:
        logger.warning(f"No text files found in {input_dir}")
        return

    logger.info(f"Found {len(file_data)} text files to process")

    # Create a wrapper function for batch processing
    def batch_processor(batch, batch_idx, **kwargs):
        return process_qa_batch(batch, batch_idx, qa_prompt, debug)

    # Process files in batches and collect all Q&A pairs
    all_qa_pairs = []
    for batch_qa_pairs in process_batches(
        file_data, BATCH_SIZE, batch_processor, description="Generating Q&A Pairs"
    ):
        all_qa_pairs.extend(batch_qa_pairs)

    # Save all Q&A pairs to output file
    if all_qa_pairs:
        output_path = output_dir / "all_qa_pairs.jsonl"
        with output_path.open("w", encoding="utf-8") as out_f:
            for qa_pair in all_qa_pairs:
                out_f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
        logger.info(
            f"✅ Q&A generation completed. Saved {len(all_qa_pairs)} Q&A pairs to {output_path}"
        )
    else:
        logger.warning("No Q&A pairs generated from any file.")


def generate_qa_pairs_with_validation(
    input_dir: Path,
    output_dir: Path,
    prompts_path: str,
    debug: bool = False,
    max_retries: int = 2,
):
    """
    Generate Q&A pairs with validation and retry logic.
    """
    logger.info(f"Generating Q&A pairs from files in {input_dir} with validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)
    qa_prompt = prompts["qa_prompt"]

    # Load all text files
    file_data = load_text_files(input_dir)

    if not file_data:
        logger.warning(f"No text files found in {input_dir}")
        return

    logger.info(f"Found {len(file_data)} text files to process")

    # Process files with validation
    all_qa_pairs = []
    for i in range(0, len(file_data), BATCH_SIZE):
        batch = file_data[i : i + BATCH_SIZE]
        batch_idx = i // BATCH_SIZE

        batch_qa_pairs = process_batch_with_validation(
            batch,
            batch_idx,
            lambda b, idx, **kwargs: process_qa_batch(b, idx, qa_prompt, debug),
            validation_function=validate_qa_result,
            max_retries=max_retries,
        )

        if batch_qa_pairs:
            all_qa_pairs.extend(batch_qa_pairs)

    # Save all Q&A pairs to output file
    if all_qa_pairs:
        output_path = output_dir / "all_qa_pairs.jsonl"
        with output_path.open("w", encoding="utf-8") as out_f:
            for qa_pair in all_qa_pairs:
                out_f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
        logger.info(
            f"✅ Q&A generation with validation completed. Saved {len(all_qa_pairs)} Q&A pairs to {output_path}"
        )
    else:
        logger.warning("No Q&A pairs generated from any file.")
