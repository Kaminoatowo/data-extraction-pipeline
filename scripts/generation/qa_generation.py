import json
from pathlib import Path
import re
from config.config import openai_client, MODEL_NAME
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts
from scripts.utils.content_format import generate_fine_tuning
from scripts.utils.batch_processor import process_batches
from itertools import islice
from multiprocessing import Pool

logger = setup_logger("qa_generation")

# Batch processing configuration
BATCH_SIZE = 2  # Process multiple files at once


def batched(iterable, n):
    """Yield successive n-sized batches from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


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


def generate_qa(
    file_path: Path, text_content: str, qa_prompt: str, debug: bool
) -> list:
    """
    Wrapper function for parallel Q&A generation using multiprocessing.

    Args:
        file_path: Path to the file being processed
        text_content: Content of the text file
        qa_prompt: The Q&A generation prompt
        debug: Whether debug mode is enabled

    Returns:
        List of Q&A pairs with file information
    """
    logger.debug(f"Generating Q&A for file: {file_path.name}")

    qa_pairs = generate_qa_from_text(text_content, qa_prompt, debug)

    if qa_pairs:
        logger.info(f"Extracted {len(qa_pairs)} Q&A pairs from {file_path.name}")
        return qa_pairs
    else:
        logger.warning(f"No Q&A generated for {file_path.name}")
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
    input_dir: Path,
    output_dir: Path,
    prompts_path: str,
    debug: bool = False,
    parallel_batch_size: int = 10,
):
    """
    Generates Q&A pairs from each RAG or CPT text file in input_dir using parallel processing.
    """
    logger.info(
        f"Generating Q&A pairs from files in {input_dir} using parallel processing"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)
    qa_prompt = prompts["qa_prompt"]

    # Load all text files
    file_data = load_text_files(input_dir)

    if not file_data:
        logger.warning(f"No text files found in {input_dir}")
        return

    logger.info(f"Found {len(file_data)} text files to process")

    # Prepare tasks for parallel processing
    tasks = [
        (file_path, text_content, qa_prompt, debug)
        for file_path, text_content in file_data
    ]

    all_qa_pairs = []

    if debug:
        logger.info(
            "Debug mode enabled, processing files sequentially without parallelization"
        )
        for file_path, text_content, qa_prompt, debug in tasks:
            qa_pairs = generate_qa(file_path, text_content, qa_prompt, debug)
            all_qa_pairs.extend(qa_pairs)
    else:
        # Use multiprocessing for parallel execution
        pool = Pool()
        try:
            for batch in batched(tasks, parallel_batch_size):
                logger.info(f"Processing batch of {len(batch)} files in parallel")
                batch_results = pool.starmap(generate_qa, batch)

                # Flatten results from the batch
                for qa_pairs in batch_results:
                    all_qa_pairs.extend(qa_pairs)
        finally:
            pool.close()
            pool.join()

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
    parallel_batch_size: int = 10,
):
    """
    Generate Q&A pairs with validation, retry logic, and parallel processing.
    """
    logger.info(
        f"Generating Q&A pairs from files in {input_dir} with validation and parallel processing"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)
    qa_prompt = prompts["qa_prompt"]

    # Load all text files
    file_data = load_text_files(input_dir)

    if not file_data:
        logger.warning(f"No text files found in {input_dir}")
        return

    logger.info(f"Found {len(file_data)} text files to process")

    def process_single_file_with_validation(
        file_path: Path, text_content: str, qa_prompt: str, debug: bool
    ):
        """Process a single file with validation and retry logic."""
        for attempt in range(max_retries):
            try:
                qa_pairs = generate_qa(file_path, text_content, qa_prompt, debug)

                if validate_qa_result(qa_pairs):
                    return qa_pairs
                else:
                    logger.warning(
                        f"File {file_path.name} validation failed on attempt {attempt + 1}"
                    )
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error(
                            f"File {file_path.name} failed validation after {max_retries} attempts"
                        )
                        return []
            except Exception as e:
                logger.error(
                    f"File {file_path.name} failed on attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"File {file_path.name} failed after {max_retries} attempts"
                    )
                    return []
        return []

    # Prepare tasks for parallel processing
    tasks = [
        (file_path, text_content, qa_prompt, debug)
        for file_path, text_content in file_data
    ]

    all_qa_pairs = []

    if debug:
        logger.info(
            "Debug mode enabled, processing files sequentially without parallelization"
        )
        for file_path, text_content, qa_prompt, debug in tasks:
            qa_pairs = process_single_file_with_validation(
                file_path, text_content, qa_prompt, debug
            )
            all_qa_pairs.extend(qa_pairs)
    else:
        # Use multiprocessing for parallel execution
        pool = Pool()
        try:
            for batch in batched(tasks, parallel_batch_size):
                logger.info(
                    f"Processing batch of {len(batch)} files in parallel with validation"
                )
                batch_results = pool.starmap(process_single_file_with_validation, batch)

                # Flatten results from the batch
                for qa_pairs in batch_results:
                    all_qa_pairs.extend(qa_pairs)
        finally:
            pool.close()
            pool.join()

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
