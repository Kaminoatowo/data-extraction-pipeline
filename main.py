import argparse
from pathlib import Path
from config.config import DEBUG_MODE

from scripts.utils.logger import setup_logger
from scripts.pdf.pdf_splitter import run_pdf_split
from scripts.ocr.ocr_runner import batch_ocr
from scripts.generation.cpt_rag_generation import generate_outputs_from_ocr_txt
from scripts.equations.equation_extraction import generate_equation_jsons
from scripts.equations.equation_formatting import format_equation_conversations
from scripts.generation.qa_generation import generate_qa_pairs
from scripts.generation.synthetic_data import generate_synthetic_data

logger = setup_logger("pipeline")


def run_pipeline(args):
    # Handle debug mode configuration
    if args.debug_mode:
        DEBUG_MODE = True
        logger.info("Debug mode enabled: %s", DEBUG_MODE)

    work_dir = Path(args.work_dir)
    split_output_dir = work_dir / "pdfs"
    split_output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_pdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting pipeline with input PDF: %s", input_path)

    if args.split_only:
        logger.info("Running split-only mode.")
        run_pdf_split(input_path, split_output_dir, args.pages_per_chunk)
        return

    ocr_output_dir = work_dir / "ocr_output"

    # Step 1: OCR / PDF Text Extraction
    if args.run_ocr:
        logger.info("Running OCR...")
        pdf_chunks = sorted(split_output_dir.glob("*.pdf"))

        if not pdf_chunks:
            logger.warning(
                "No PDF chunks found in %s. Make sure splitting has been done.",
                split_output_dir,
            )
            return

        ocr_txt_paths = batch_ocr(pdf_chunks, ocr_output_dir, debug=DEBUG_MODE)

        if ocr_txt_paths:
            logger.info("OCR completed. Text files saved to: %s", ocr_output_dir)
        else:
            logger.error("OCR returned no output. Check model or PDF inputs.")
            return
    else:
        ocr_txt_files = sorted(ocr_output_dir.glob("*.txt"))

        if ocr_txt_files:
            logger.info("Loading existing OCR outputs from %s", ocr_output_dir)
            combined_text = "\n\n".join(
                f"# {txt_file.name}\n{txt_file.read_text(encoding='utf-8')}"
                for txt_file in ocr_txt_files
            )
            text_path = output_dir / "extracted_text.txt"
            text_path.write_text(combined_text, encoding="utf-8")
            logger.info("Combined text saved to %s", text_path)
        else:
            logger.error(
                "No existing OCR text files found in %s. Run OCR first or provide input.",
                ocr_output_dir,
            )
            return

    # Step 3: RAG Data
    if args.run_rag:
        logger.info("Running CPT/RAG generation...")
        generate_outputs_from_ocr_txt(
            ocr_txt_dir=ocr_output_dir,
            output_rag_dir=output_dir / "output_rag",
            output_cpt_dir=output_dir / "output_cpt",
            prompts_path=Path("config/prompts.yaml"),
            debug=DEBUG_MODE,  # or wherever your prompts live
        )
        logger.info(
            "RAG and CPT outputs generated and saved to %s",
            output_dir / "output_rag",
        )

    if args.run_equations:
        logger.info("Running equation extraction...")
        generate_equation_jsons(
            ocr_txt_dir=ocr_output_dir,
            output_dir=output_dir / "equations_json",
            prompts_path=Path("config/prompts.yaml"),
            debug=DEBUG_MODE,
        )
        logger.info(
            "Equation JSONs generated and saved to %s",
            output_dir / "equations_json",
        )

        json_dir = output_dir / "equations_json"
        formatted_dir = output_dir / "formatted_equations"
        for json_file in json_dir.glob("*.json"):
            format_equation_conversations(
                json_file, formatted_dir / f"{json_file.stem}.json"
            )
        logger.info("Formatted equations saved to %s", formatted_dir)

    if args.run_synth:
        logger.info("Starting synthetic data generation...")
        generate_synthetic_data(
            input_folder=output_dir / "output_rag",
            output_file=output_dir / "synthetic_data.jsonl",
            prompts_path=Path("config/prompts.yaml"),
            debug=DEBUG_MODE,
        )
        logger.info("Synthetic data generation complete.")

    if args.run_qa:
        generate_qa_pairs(
            input_dir=output_dir / "output_rag",  # or output_cpt
            output_dir=output_dir / "qa_pairs",
            prompts_path=Path("config/prompts.yaml"),
            debug=DEBUG_MODE,
        )

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Pipeline from PDF to QA.")
    parser.add_argument(
        "--input_pdf", type=str, required=True, help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/outputs",
        help="Directory to store outputs.",
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default="data",
        help="Base directory for all output files",
    )

    # Flags for partial execution
    parser.add_argument("--split_only", action="store_true", help="Only split PDF")
    parser.add_argument(
        "--run_split", action="store_true", help="Run PDF splitting step before OCR"
    )
    parser.add_argument(
        "--pages_per_chunk",
        type=int,
        default=10,
        help="Number of pages per split chunk",
    )

    parser.add_argument(
        "--run_ocr", action="store_true", help="Run OCR to extract text from PDF."
    )
    #    parser.add_argument("--run_split", action="store_true", help="Run text splitting.")
    parser.add_argument(
        "--run_rag", action="store_true", help="Run RAG data generation."
    )
    parser.add_argument("--run_cpt", action="store_true", help="Run CPT generation.")
    parser.add_argument(
        "--run_equations", action="store_true", help="Run equation extraction."
    )
    parser.add_argument(
        "--run_qa", action="store_true", help="Run question generation."
    )

    parser.add_argument(
        "--run_synth", action="store_true", help="Run synthetic data generation."
    )

    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode. When enabled, disables gpt APIs.",
    )

    parser.add_argument("--run_all", action="store_true", help="Run the full pipeline.")

    args = parser.parse_args()

    if args.run_all:
        args.run_ocr = True
        args.run_split = True
        args.run_rag = True
        args.run_cpt = True
        args.run_equations = True
        args.run_qa = True

    run_pipeline(args)
