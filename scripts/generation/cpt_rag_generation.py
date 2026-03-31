import json
import os
from pathlib import Path
from config.config import openai_client, MODEL_NAME
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts  # adjust path as needed
from scripts.utils.content_format import generate_pretraining
from itertools import islice
from multiprocessing import Pool


logger = setup_logger("cpt_rag_generation")


def batched(iterable, n):
    """Yield successive n-sized batches from iterable."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def load_txt_files(txt_folder: Path) -> dict:
    """
    Loads all .txt files from a directory and returns a dict of contents.
    """
    logger.info(f"Loading OCR text files from: {txt_folder}")
    txt_files = sorted(txt_folder.glob("*.txt"))
    txt_contents = {}
    for i, txt_file in enumerate(txt_files):
        with open(txt_file, "r", encoding="utf-8") as f:
            txt_contents[i] = f.read()
            logger.debug(f"Loaded text file: {txt_file}")
    return txt_contents


def extract_cpt(message):
    content = []
    for text in message.split("##"):
        text = text.strip().replace("\n", " ").replace("\\", "\\\\").replace('"', '\\"')
        if len(text) < 100:
            continue
        content.append(generate_pretraining(text))
    return content


def txt_folder_to_jsonl(txt_folder, output_jsonl):
    lines = []
    for filename in os.listdir(txt_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(txt_folder, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    lines = extract_cpt(content)
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for item in lines:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")


def generate_rag_output(
    text: str, rag_prompt: str, output_path: Path, debug: bool
) -> str:
    """
    Generate RAG-style output using a prompt and save to file.
    """
    logger.info(f"Generating RAG output -> {output_path.name}")
    rag_message = {"role": "system", "content": rag_prompt}
    if debug:
        logger.debug(f"RAG prompt: {rag_message['content']}")
        logger.debug(f"Input text length: {len(text)} characters")
        return "Debug mode enabled, skipping RAG generation."
    try:
        rag_response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[rag_message, {"role": "user", "content": text}],
            temperature=0.5,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = rag_response.choices[0].message.content
        output_path.write_text(content, encoding="utf-8")
        logger.debug(f"RAG output saved to {output_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to generate RAG output for {output_path.name}: {e}")
        return f"Error: {e}"


def generate_cpt_output(
    text: str, cpt_prompt: str, output_path: Path, debug: bool
) -> str:
    """
    Generate CPT-style output using a prompt and save to file.
    """
    logger.info(f"Generating CPT output -> {output_path.name}")
    cpt_message = {"role": "system", "content": cpt_prompt}
    if debug:
        logger.debug(f"CPT prompt: {cpt_message['content']}")
        logger.debug(f"Input text length: {len(text)} characters")
        return "Debug mode enabled, skipping CPT generation."
    try:
        cpt_response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[cpt_message, {"role": "user", "content": text}],
            temperature=0.5,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = cpt_response.choices[0].message.content
        output_path.write_text(content, encoding="utf-8")
        logger.debug(f"CPT output saved to {output_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to generate CPT output for {output_path.name}: {e}")
        return f"Error: {e}"


def generate_outputs_from_ocr_txt(
    ocr_txt_dir: Path,
    output_rag_dir: Path,
    output_cpt_dir: Path,
    prompts_path: Path,
    debug: bool = False,
):
    """
    Generates RAG and CPT outputs for all .txt files in the OCR output directory.
    """

    logger.info("Starting RAG and CPT generation from OCR text.")
    prompts = load_prompts(prompts_path)
    txt_files = sorted(ocr_txt_dir.glob("*.txt"))

    rag_out_path = []
    cpt_out_path = []
    read_text = []
    for txt_file in txt_files:
        base = txt_file.stem
        logger.info(f"Processing file: {base}")

        with txt_file.open("r", encoding="utf-8") as f:
            text = f.read()
            read_text.append(text)

        path = output_rag_dir / f"{base}.txt"

        if not path.exists():
            # generate_rag_output(text, prompts["rag_prompt"], rag_out_path, debug)
            rag_out_path.append(output_rag_dir / f"{base}.txt")
        else:
            logger.warning(f"RAG output already exists for {base}. Skipping.")

        path = output_cpt_dir / f"{base}.txt"

        if not path.exists():
            # generate_cpt_output(text, prompts["cpt_prompt"], cpt_out_path, debug)
            cpt_out_path.append(output_cpt_dir / f"{base}.txt")
        else:
            logger.warning(f"CPT output already exists for {base}. Skipping.")

    if not rag_out_path or not cpt_out_path:
        logger.error("No RAG or CPT output paths planned. Exiting.")
    else:
        pool = Pool()
        # Prepare tuples of (text, rag_path, cpt_path)
        tasks = list(zip(read_text, rag_out_path, cpt_out_path))
        for batch in batched(tasks, 10):
            # RAG generation
            pool.starmap(
                generate_rag_output,
                [
                    (text, prompts["rag_prompt"], rag_path, debug)
                    for text, rag_path, _ in batch
                ],
            )
            # CPT generation
            pool.starmap(
                generate_cpt_output,
                [
                    (text, prompts["cpt_prompt"], cpt_path, debug)
                    for text, _, cpt_path in batch
                ],
            )
        pool.close()
        pool.join()
