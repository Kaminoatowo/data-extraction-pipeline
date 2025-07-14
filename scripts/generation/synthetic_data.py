import os
import json
import re
import tiktoken
from config.config import openai_client, MODEL_NAME
from scripts.utils.text_splitter import (
    count_tokens,
    load_and_split_text_files,
)
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts  # adjust path as needed
from scripts.utils.content_format import generate_pretraining
from scripts.utils.batch_processor import process_batches

logger = setup_logger("synthetic_data")


BATCH_SIZE = 1
MAX_TOKENS_PER_REQUEST = 8000
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 3000

tokenizer = tiktoken.encoding_for_model(MODEL_NAME)


def build_prompt(prompt_template, batch, batch_index):
    """Build a prompt from template and batch data."""
    prompt = prompt_template
    for i, (filename, content) in enumerate(batch):
        prompt += f"Source Document {batch_index + i + 1} ({filename}):\n{content.strip()}\n\n"

    prompt += "Now generate synthetic training content by rephrasing and restructuring the information from these sources:\n"
    return prompt


def process_batch(batch, batch_idx, prompt_template, debug=False):
    """Process a single batch and return content pieces."""
    prompt = build_prompt(prompt_template, batch, batch_idx * BATCH_SIZE)
    prompt_tokens = count_tokens(prompt)

    if prompt_tokens > MAX_TOKENS_PER_REQUEST - MAX_OUTPUT_TOKENS:
        logger.warning(
            f"Skipping batch {batch_idx + 1}: too long ({prompt_tokens} tokens)."
        )
        return []

    try:
        if debug:
            logger.debug(
                f"Debug mode enabled, skipping API call for batch {batch_idx + 1}"
            )
            return []

        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        answer_text = response.choices[0].message.content.strip()
        content_pieces = extract_synthetic_content(answer_text)

        if content_pieces:
            logger.info(
                f"Batch {batch_idx + 1}: Generated {len(content_pieces)} entries"
            )
        else:
            logger.warning(f"Batch {batch_idx + 1}: No valid content extracted")

        return content_pieces

    except Exception as e:
        logger.error(f"Batch {batch_idx + 1} failed: {e}")
        return []


def extract_synthetic_content(text):
    content_pieces = []
    sections = re.split(
        r"={3,}\s*SYNTHETIC CONTENT\s*\d*\s*={3,}", text, flags=re.IGNORECASE
    )

    for section in sections[1:]:
        content = section.strip()
        if content and len(content) > 50:
            content_pieces.append(content)

    if not content_pieces:
        potential_sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        content_pieces = [s for s in potential_sections if len(s) > 100]

    return content_pieces


def save_synthetic_data(content_pieces, output_path):
    with open(output_path, "a", encoding="utf-8") as f:
        for content in content_pieces:
            json_obj = generate_pretraining(content)
            f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


def generate_synthetic_data(
    input_folder: str, output_file: str, prompts_path: str, debug: bool = False
):
    """Main function to generate synthetic data from input folder."""
    if os.path.exists(output_file):
        logger.info(f"Removing existing output file: {output_file}")
        os.remove(output_file)

    all_texts = load_and_split_text_files(input_folder, max_tokens_per_chunk=1500)
    total_content_pieces = 0
    prompt_synth = load_prompts(prompts_path)
    prompt_template = prompt_synth["synthetic_data_extraction"]

    # Create a wrapper function that matches the expected signature for process_batches
    def process_batch_wrapper(batch, batch_idx, **kwargs):
        return process_batch(batch, batch_idx, prompt_template, debug)

    # Process all batches and save results
    for content_pieces in process_batches(
        all_texts,
        BATCH_SIZE,
        process_batch_wrapper,
        description="Generating Synthetic Data",
    ):
        save_synthetic_data(content_pieces, output_file)
        total_content_pieces += len(content_pieces)

    logger.info(
        f"✅ Done. {total_content_pieces} synthetic entries written to {output_file}"
    )
