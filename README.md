# Data Extraction Pipeline

A comprehensive LLM-powered pipeline for extracting, processing, and generating training data from PDF documents. This pipeline converts PDFs into structured datasets suitable for machine learning model training through OCR, RAG generation, equation extraction, Q&A generation, and synthetic data creation.

## Features

- **PDF Processing**: Split large PDFs into manageable chunks
- **OCR Text Extraction**: Extract text from PDF documents
- **RAG Data Generation**: Create Retrieval-Augmented Generation datasets
- **Equation Extraction**: Extract and format mathematical equations
- **Q&A Generation**: Generate question-answer pairs from text
- **Synthetic Data Generation**: Create synthetic training data
- **Data Merging**: Combine datasets for continued pre-training (CPT) and fine-tuning (FT)
- **Archive & Cleanup**: Backup and clean generated files

## Prerequisites

- Python 3.8+
- Required dependencies (install via `pip install -r requirements.txt`)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in `.env` file:
   ```
   MISTRAL_API_KEY=your_mistral_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Basic Command Structure

```bash
python main.py [OPTIONS]
```

### Core Arguments

- `--input_pdf`: Path to the input PDF file
- `--output_dir`: Directory to store outputs (default: `data/outputs`)
- `--work_dir`: Base directory for all output files (default: `data`)
- `--input_cpt`: Path to existing CPT file for merging
- `--input_ft`: Path to existing fine-tuning file for merging

### Processing Options

#### PDF Processing
- `--split_only`: Only split PDF into chunks
- `--run_split`: Run PDF splitting step before OCR
- `--pages_per_chunk`: Number of pages per split chunk (default: 10)

#### Text Extraction
- `--run_ocr`: Run OCR to extract text from PDF

#### Data Generation
- `--run_rag`: Run RAG data generation
- `--run_cpt`: Run CPT generation
- `--run_equations`: Run equation extraction
- `--run_qa`: Run question generation
- `--run_synth`: Run synthetic data generation

#### Dataset Creation
- `--gen_cpt`: Generate CPT data from RAG outputs
- `--gen_ft`: Generate fine-tuning data from RAG outputs

### Utility Options

- `--debug_mode`: Enable debug mode (disables GPT APIs)
- `--clean_all`: Delete all generated files (prompts for confirmation)
- `--archive_files`: Archive all generated files to timestamped zip
- `--archive_dir`: Directory to store archive files (default: `work_dir/archive`)
- `--run_all`: Run the complete pipeline

## Usage Examples

### Complete Pipeline
Run the entire pipeline from PDF to training datasets:
```bash
python main.py --input_pdf "document.pdf" --run_all
```

### Split PDF Only
Split a large PDF into smaller chunks:
```bash
python main.py --input_pdf "large_document.pdf" --split_only --pages_per_chunk 5
```

### OCR Processing
Extract text from PDF chunks:
```bash
python main.py --input_pdf "document.pdf" --run_split --run_ocr
```

### Generate Training Data
Create RAG datasets and Q&A pairs:
```bash
python main.py --input_pdf "document.pdf" --run_rag --run_qa --run_equations
```

### Create Final Datasets
Generate merged datasets for training:
```bash
python main.py --input_pdf "document.pdf" --gen_cpt --gen_ft
```

### Archive Files
Create a backup of all generated files:
```bash
python main.py --input_pdf "document.pdf" --archive_files
```

### Clean Up
Delete all generated files (with optional archiving):
```bash
python main.py --input_pdf "document.pdf" --clean_all
```

### Debug Mode
Run pipeline without API calls for testing:
```bash
python main.py --input_pdf "document.pdf" --run_all --debug_mode
```

### Merge with Existing Data
Combine new datasets with existing training data:
```bash
python main.py --input_pdf "new_document.pdf" --gen_cpt --gen_ft \
  --input_cpt "existing_cpt.jsonl" --input_ft "existing_ft.jsonl"
```

## Output Structure

The pipeline generates files in the following directory structure:

```
data/
├── pdfs/                    # Split PDF chunks
├── ocr_output/             # OCR extracted text files
├── texts/                  # Processed text files
├── datasets/               # Raw equation datasets
├── jsonl_files/            # Final merged datasets
│   ├── new_pretraining.jsonl
│   ├── new_finetuning.jsonl
│   ├── merged_cpt.jsonl
│   └── merged_finetuning.jsonl
├── archive/                # Archived files
└── outputs/
    ├── output_rag/         # RAG generated data
    ├── output_cpt/         # CPT data
    ├── formatted_equations/ # Formatted equation datasets
    ├── qa_pairs/           # Generated Q&A pairs
    ├── extracted_text.txt  # Combined OCR text
    └── synthetic_data.jsonl # Synthetic training data
```

## Configuration

The pipeline uses configuration files in the `config/` directory:

- `config.py`: Main configuration settings
- `prompts.yaml`: Prompt templates for LLM generation

## File Management

### Archiving
The archive function creates timestamped zip files containing all generated files:
- Format: `pipeline_backup_YYYYMMDD_HHMMSS.zip`
- Includes: `.pdf`, `.txt`, `.json`, `.jsonl` files
- Preserves directory structure

### Cleanup
The cleanup function removes generated files while preserving directory structure:
- Prompts for confirmation before deletion
- Offers optional archiving before cleanup
- Only removes specified file types

## Debug Mode

When `--debug_mode` is enabled:
- API calls to GPT models are disabled
- Useful for testing pipeline logic
- Generates placeholder outputs for development

## Tips

1. **Start Small**: Test with a small PDF first using `--split_only`
2. **Check Outputs**: Verify each step before proceeding to the next
3. **Use Archives**: Always archive important datasets before cleanup
4. **Monitor Resources**: Large PDFs may require significant processing time
5. **Debug First**: Use `--debug_mode` to test configuration changes

## Troubleshooting

- **No PDF chunks found**: Ensure PDF splitting completed successfully
- **OCR failed**: Check PDF quality and OCR model configuration
- **API errors**: Verify API keys are correctly set in environment
- **Memory issues**: Process smaller chunks or reduce batch sizes

## Contributing

1. Follow the existing code structure
2. Add logging for new features
3. Update this README for new functionality
4. Test with debug mode before production use

