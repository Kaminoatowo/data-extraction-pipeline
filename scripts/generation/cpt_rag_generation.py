import os
import glob
from pathlib import Path
from openai import OpenAI
from config.config import openai_client
from scripts.utils.logger import setup_logger
from scripts.utils.prompt_loader import load_prompts  # adjust path as needed

logger = setup_logger("ocr_runner")


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
    rag_response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[rag_message, {"role": "user", "content": text}],
        temperature=0.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = rag_response.choices[0].message.content
    output_path.write_text(content, encoding="utf-8")
    logger.debug(f"RAG output saved to {output_path}")
    return content


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
    cpt_response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[cpt_message, {"role": "user", "content": text}],
        temperature=0.5,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = cpt_response.choices[0].message.content
    output_path.write_text(content, encoding="utf-8")
    logger.debug(f"CPT output saved to {output_path}")
    return content


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

    for txt_file in txt_files:
        base = txt_file.stem
        logger.info(f"Processing file: {base}")

        with txt_file.open("r", encoding="utf-8") as f:
            text = f.read()

        rag_out_path = output_rag_dir / f"{base}.txt"
        cpt_out_path = output_cpt_dir / f"{base}.txt"

        if not rag_out_path.exists():
            generate_rag_output(text, prompts["rag_prompt"], rag_out_path, debug)
        else:
            logger.warning(f"RAG output already exists for {base}. Skipping.")

        if not cpt_out_path.exists():
            generate_cpt_output(text, prompts["cpt_prompt"], cpt_out_path, debug)
        else:
            logger.warning(f"CPT output already exists for {base}. Skipping.")
