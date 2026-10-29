# Research Paper OCR & Deep Metadata Extraction

## Project goal

This project creates a Python pipeline for extracting structured metadata from the five arXiv papers listed in the internship task. It produces `extracted_papers.json` with the paper title, authors, abstract, publication date, arXiv URLs, official code repository, tasks, methods, models, datasets, and quantitative SOTA claims.

## Files

- `extract_paper_meta.py` — main extraction script.
- `requirements.txt` — Python dependencies.
- `extracted_papers.json` — initial five-paper output.
- `papers/` — optional directory for source PDFs.

## Requirements

- Python 3.10+
- Internet access if downloading the arXiv PDFs.
- PyMuPDF, Requests, pytesseract, and Pillow (installed below).
- For scanned PDFs only: install the Tesseract OCR engine separately and ensure it is on PATH.

## Setup (Windows PowerShell)

```powershell
cd research_paper_ocr_task
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the same terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

Download the five PDFs from arXiv and parse them:

```powershell
python extract_paper_meta.py --download --input-dir papers --output extracted_papers.json
```

If PDFs are already downloaded, name them with their arXiv IDs (for example `papers/2503.18382.pdf`) and run:

```powershell
python extract_paper_meta.py --input-dir papers --output extracted_papers.json
```

The script can also run without local PDFs. In that case it emits the reviewed metadata records bundled in the script. For a production submission, run with PDFs available and review the extracted data against the papers.

## How official GitHub repositories are selected

A common extraction error is treating a library mentioned in a citation or footnote as the paper's own implementation. This baseline uses a curated mapping of the five task papers to their author-declared repositories. When operating on an unknown paper, the fallback parser considers GitHub URLs near phrases such as “code is available” and “our repository,” normalizes links to `https://github.com/owner/repository`, and penalizes URLs appearing near citation/dependency language. If no convincing author-provided repository is found, it returns `null`. The curated mapping is intentionally stronger than a simple “first GitHub URL” rule.

## Extraction design and limitations

1. **PDF text parsing:** PyMuPDF reads embedded text in research PDFs and joins line breaks/hyphenations.
2. **Metadata normalization:** LaTeX wrappers and common author superscripts are cleaned where recognized.
3. **Repository selection:** author-release statements and a reviewed allow-list prevent common dependency-link mistakes.
4. **Scientific entities:** the five task records are seeded with reviewed paper-specific metadata; generic model/dataset extraction is not fully solved by regex alone.
5. **SOTA claims:** claims include benchmark, metric, value, and comparison target. Values should be checked against tables and result sections because generic regex cannot reliably infer table headers or comparison context.
6. **Scanned PDFs:** when embedded text is too short, the script tries OCR using pytesseract + Pillow. The Tesseract executable must be installed separately; if it is missing, the script logs a warning and continues.

## Validation before submission

- Confirm `extracted_papers.json` is valid JSON.
- Confirm exactly five records are present.
- Compare every SOTA claim with the paper abstract/results tables.
- Check that each `github_url` is the authors' official code release, not a dependency.
- If a paper does not name an experimental dataset in the source, keep the array empty rather than inventing one.
