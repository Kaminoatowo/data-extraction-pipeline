import argparse
from pathlib import Path

from scripts.utils.logger import setup_logger
from scripts.pdf.pdf_splitter import run_pdf_split
from scripts.ocr.pdf_to_text import extract_text_from_pdf
from scripts.splitter.splitter import split_text
from scripts.rag.rag_builder import build_rag_data
from scripts.cpt.cpt_generator import generate_cpt
from scripts.equations.extract_equations import extract_equations
from scripts.qa.qa_generator import generate_qa_pairs

logger = setup_logger("pipeline")


def run_pipeline(args):
    input_pdf = Path(args.input_pdf)
    work_dir = Path(args.work_dir)
    split_output_dir = work_dir / "pdf_chunks"
    split_output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_pdf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting pipeline with input PDF: %s", input_path)

    if args.split_only:
        logger.info("Running split-only mode.")
        run_pdf_split(input_pdf, split_output_dir, args.pages_per_chunk)
        return

    # Step 1: OCR / PDF Text Extraction
    if args.run_ocr:
        logger.info("Running OCR...")
        text = extract_text_from_pdf(input_path)
        text_path = output_dir / "extracted_text.txt"
        text_path.write_text(text, encoding="utf-8")
        logger.info("Text extracted and saved to %s", text_path)
    else:
        text_path = output_dir / "extracted_text.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
            logger.info("Loaded existing extracted text from %s", text_path)
        else:
            logger.error("Text file not found. Run OCR or provide text manually.")
            return

    # Step 2: Split Text
    if args.run_split:
        logger.info("Splitting text...")
        splits = split_text(text)
        split_path = output_dir / "splits.json"
        import json

        split_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False))
        logger.info("Text splits saved to %s", split_path)
    else:
        split_path = output_dir / "splits.json"
        if split_path.exists():
            import json

            splits = json.loads(split_path.read_text(encoding="utf-8"))
            logger.info("Loaded existing splits from %s", split_path)
        else:
            logger.error("Splits not found. Run text splitter or provide manually.")
            return

    # Step 3: RAG Data
    if args.run_rag:
        logger.info("Generating RAG data...")
        rag_data = build_rag_data(splits, output_dir)
        logger.info("RAG data generated.")
    else:
        logger.info("Skipping RAG generation.")
        rag_data = None

    # Step 4: CPT
    if args.run_cpt:
        logger.info("Generating CPT...")
        generate_cpt(splits, output_dir)
        logger.info("CPT generation complete.")
    else:
        logger.info("Skipping CPT generation.")

    # Step 5: Equation Extraction
    if args.run_equations:
        logger.info("Extracting equations...")
        extract_equations(splits, output_dir)
        logger.info("Equation extraction complete.")
    else:
        logger.info("Skipping equation extraction.")

    # Step 6: QA Generation
    if args.run_qa:
        logger.info("Generating Q&A...")
        generate_qa_pairs(splits, output_dir)
        logger.info("Q&A generation complete.")
    else:
        logger.info("Skipping Q&A generation.")

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Pipeline from PDF to QA.")
    parser.add_argument(
        "--input_pdf", type=str, required=True, help="Path to the input PDF file."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to store outputs."
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        default="workspace",
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
    parser.add_argument("--run_split", action="store_true", help="Run text splitting.")
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
