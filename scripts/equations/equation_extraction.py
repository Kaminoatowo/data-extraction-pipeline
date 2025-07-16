import json
from pathlib import Path
from config.config import openai_client, MODEL_NAME
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts
from scripts.utils.batch_processor import process_batches
from itertools import islice
from multiprocessing import Pool

logger = setup_logger("equation_extraction")

# Batch processing configuration
BATCH_SIZE = 3  # Process multiple files at once


def batched(iterable, n):
    """Yield successive n-sized batches from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def process_equation_batch(batch, batch_idx, prompt, output_dir, debug=False):
    """
    Process a batch of text files for equation extraction.

    Args:
        batch: List of (file_path, file_content) tuples
        batch_idx: Index of the current batch
        prompt: The equation extraction prompt
        output_dir: Directory to save JSON files
        debug: Whether debug mode is enabled

    Returns:
        List of successfully processed file paths
    """
    processed_files = []

    for file_path, text_content in batch:
        logger.debug(f"Processing file: {file_path.name}")

        result = extract_equations_from_text(text_content, prompt, debug)

        if result:
            out_path = output_dir / f"{file_path.stem}.json"
            with out_path.open("w", encoding="utf-8") as json_file:
                json.dump(result, json_file, indent=2, ensure_ascii=False)
            logger.info(f"Saved equation JSON to {out_path}")
            processed_files.append(file_path)
        else:
            logger.warning(f"No equations extracted from {file_path}")

    return processed_files


def load_text_files(input_txt: Path):
    """
    Load all text files and return list of (file_path, content) tuples.

    Args:
        input_txt: Directory containing text files

    Returns:
        List of (file_path, content) tuples
    """
    txt_files = sorted(input_txt.glob("*.txt"))
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


def extract_equations(
    file_path: Path, text_content: str, prompt: str, output_dir: Path, debug: bool
) -> bool:
    """
    Wrapper function for parallel equation extraction using multiprocessing.

    Args:
        file_path: Path to the file being processed
        text_content: Content of the text file
        prompt: The equation extraction prompt
        output_dir: Directory to save JSON files
        debug: Whether debug mode is enabled

    Returns:
        bool: True if extraction was successful, False otherwise
    """
    logger.debug(f"Processing file: {file_path.name}")

    result = extract_equations_from_text(text_content, prompt, debug)

    if result:
        out_path = output_dir / f"{file_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as json_file:
            json.dump(result, json_file, indent=2, ensure_ascii=False)
        logger.info(f"Saved equation JSON to {out_path}")
        return True
    else:
        logger.warning(f"No equations extracted from {file_path}")
        return False


def validate_equation_result(result):
    """
    Validate the equation extraction result.

    Args:
        result: List of file paths that were processed successfully

    Returns:
        bool: True if the result is valid, False otherwise
    """
    if not result:
        return False

    # Check if at least one file was processed successfully
    return len(result) > 0


def generate_equation_jsons(
    input_txt: Path,
    output_dir: Path,
    prompts_path: str,
    debug: bool = False,
    parallel_batch_size: int = 10,
):
    """
    Process all OCR .txt files and generate JSON files with extracted equations using parallel processing.
    """
    logger.info(f"Generating equation JSONs from {input_txt} using parallel processing")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)
    prompt = prompts["equation_extraction_prompt"]

    # Load all text files
    file_data = load_text_files(input_txt)

    if not file_data:
        logger.warning(f"No text files found in {input_txt}")
        return

    logger.info(f"Found {len(file_data)} text files to process")

    # Prepare tasks for parallel processing
    tasks = [
        (file_path, text_content, prompt, output_dir, debug)
        for file_path, text_content in file_data
    ]

    total_processed = 0

    if debug:
        logger.info(
            "Debug mode enabled, processing files sequentially without parallelization"
        )
        for file_path, text_content, prompt, output_dir, debug in tasks:
            success = extract_equations(
                file_path, text_content, prompt, output_dir, debug
            )
            if success:
                total_processed += 1
    else:
        # Use multiprocessing for parallel execution
        pool = Pool()
        try:
            for batch in batched(tasks, parallel_batch_size):
                logger.info(f"Processing batch of {len(batch)} files in parallel")
                batch_results = pool.starmap(extract_equations, batch)

                # Count successful extractions
                total_processed += sum(1 for success in batch_results if success)
        finally:
            pool.close()
            pool.join()

    logger.info(
        f"✅ Equation extraction completed. Processed {total_processed} files successfully."
    )


def generate_equation_jsons_with_validation(
    input_txt: Path,
    output_dir: Path,
    prompts_path: str,
    debug: bool = False,
    max_retries: int = 2,
    parallel_batch_size: int = 10,
):
    """
    Process all OCR .txt files with validation, retry logic, and parallel processing.
    """
    logger.info(
        f"Generating equation JSONs from {input_txt} with validation and parallel processing"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompts_path)
    prompt = prompts["equation_extraction_prompt"]

    # Load all text files
    file_data = load_text_files(input_txt)

    if not file_data:
        logger.warning(f"No text files found in {input_txt}")
        return

    logger.info(f"Found {len(file_data)} text files to process")

    def process_single_file_with_validation(
        file_path: Path, text_content: str, prompt: str, output_dir: Path, debug: bool
    ):
        """Process a single file with validation and retry logic."""
        for attempt in range(max_retries):
            try:
                success = extract_equations(
                    file_path, text_content, prompt, output_dir, debug
                )

                if success:
                    return True
                else:
                    logger.warning(
                        f"File {file_path.name} extraction failed on attempt {attempt + 1}"
                    )
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error(
                            f"File {file_path.name} failed extraction after {max_retries} attempts"
                        )
                        return False
            except Exception as e:
                logger.error(
                    f"File {file_path.name} failed on attempt {attempt + 1}: {e}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"File {file_path.name} failed after {max_retries} attempts"
                    )
                    return False
        return False

    # Prepare tasks for parallel processing
    tasks = [
        (file_path, text_content, prompt, output_dir, debug)
        for file_path, text_content in file_data
    ]

    total_processed = 0

    if debug:
        logger.info(
            "Debug mode enabled, processing files sequentially without parallelization"
        )
        for file_path, text_content, prompt, output_dir, debug in tasks:
            success = process_single_file_with_validation(
                file_path, text_content, prompt, output_dir, debug
            )
            if success:
                total_processed += 1
    else:
        # Use multiprocessing for parallel execution
        pool = Pool()
        try:
            for batch in batched(tasks, parallel_batch_size):
                logger.info(
                    f"Processing batch of {len(batch)} files in parallel with validation"
                )
                batch_results = pool.starmap(process_single_file_with_validation, batch)

                # Count successful extractions
                total_processed += sum(1 for success in batch_results if success)
        finally:
            pool.close()
            pool.join()

    logger.info(
        f"✅ Equation extraction with validation completed. Processed {total_processed} files successfully."
    )
