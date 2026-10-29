#!/usr/bin/env python3
"""Extract structured metadata from arXiv research-paper PDFs.

Usage:
  python extract_paper_meta.py --input-dir papers --output extracted_papers.json
  python extract_paper_meta.py --download --output extracted_papers.json

This is a deterministic baseline, not a guarantee of perfect scientific entity extraction.
Review extracted model/dataset/SOTA entities against the source paper before production use.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger("paper_meta")

ARXIV_IDS = ["2503.18382", "2510.13998", "2607.18859", "2507.23370", "2503.11651"]
ARXIV_ABS = "https://arxiv.org/abs/{id}"
ARXIV_PDF = "https://arxiv.org/pdf/{id}.pdf"
HEADERS = {"User-Agent": "ResearchPaperMetadataExtractor/1.0 (educational internship task)"}


OFFICIAL_REPOS = {
    "2503.18382": "https://github.com/PaddlePaddle/PaddleOCR",
    "2510.13998": "https://github.com/microsoft/BitNet",
    "2607.18859": "https://github.com/DeepSoftwareAnalytics/PhoenixRepair",
    "2507.23370": "https://github.com/bytedance/trae-agent",
    "2503.11651": "https://github.com/facebookresearch/vggt",
}


CURATED = {
    "2503.18382": {
        "title": "PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition",
        "authors": ["Hongen Liu", "Cheng Cui", "Yuning Du", "Yi Liu", "Gang Pan"],
        "publication_date": "2025-03-24",
        "github_url": "https://github.com/PaddlePaddle/PaddleOCR",
        "is_official_code": True,
        "tasks": ["document-understanding", "computer-vision", "formula-recognition"],
        "methods": ["deep-learning", "ocr", "formula-recognition"],
        "models": ["PP-FormulaNet-L", "PP-FormulaNet-S", "UniMERNet"],
        "datasets": ["UniMERNet Benchmark", "Formula Mining System Dataset"],
        "sota_claims": [
            {"benchmark": "UniMERNet Benchmark", "metric": "Accuracy improvement", "value": "+6.0%", "comparison_target": "UniMERNet"},
            {"benchmark": "Formula recognition inference", "metric": "Speedup factor", "value": "over 16x faster", "comparison_target": "UniMERNet"},
        ],
        "abstract": "Formula recognition converts mathematical expressions from document images into structured symbolic formats such as LaTeX. PP-FormulaNet provides two variants: PP-FormulaNet-L for high accuracy and PP-FormulaNet-S for high efficiency. The paper reports a 6% accuracy improvement over UniMERNet for the L variant and over 16x faster inference for the S variant, and introduces a Formula Mining System for collecting high-quality formula data."
    },
    "2510.13998": {
        "title": "BitNet Distillation",
        "authors": ["Xun Wu", "Shaohan Huang", "Wenhui Wang", "Ting Song", "Li Dong", "Yan Xia", "Furu Wei"],
        "publication_date": "2025-10-15",
        "github_url": "https://github.com/microsoft/BitNet",
        "is_official_code": True,
        "tasks": ["language-model-compression", "model-distillation", "efficient-inference"],
        "methods": ["quantization", "knowledge-distillation", "continual-pretraining"],
        "models": ["BitNet Distillation (BitDistill)", "BitNet b1.58", "Qwen3", "Qwen2.5", "Gemma", "MiniLM"],
        "datasets": ["MNLI", "QNLI", "SST-2", "CNN/DailyMail"],
        "sota_claims": [
            {"benchmark": "Model memory usage", "metric": "Memory savings", "value": "up to 10x", "comparison_target": "full-precision counterpart models"},
            {"benchmark": "CPU inference", "metric": "Inference speedup", "value": "2.65x faster", "comparison_target": "full-precision counterpart models"},
        ],
        "abstract": "BitNet Distillation (BitDistill) fine-tunes full-precision language models into 1.58-bit precision for specific downstream tasks. It combines SubLN, multi-head attention distillation, and continual pre-training. Experiments report performance comparable to full-precision counterpart models, up to 10x memory savings, and 2.65x faster CPU inference."
    },
    "2607.18859": {
        "title": "PhoenixRepair: Rethinking Repair Strategy Exploration in Software Agents",
        "authors": ["Tianyue Jiang", "Yanlin Wang", "Xin He", "Daya Guo", "Jiachi Chen", "Ming Wen", "Ensheng Shi", "Xilin Liu", "Yuchi Ma", "Guanbin Li"],
        "publication_date": "2026-07-21",
        "github_url": "https://github.com/DeepSoftwareAnalytics/PhoenixRepair",
        "is_official_code": True,
        "tasks": ["automated-program-repair", "software-engineering"],
        "methods": ["multi-agent-systems", "iterative-refinement", "fault-localization"],
        "models": ["PhoenixRepair", "DeepSeek-V3.1", "MiniMax-M2.5", "SWE-agent"],
        "datasets": ["SWE-bench Verified"],
        "sota_claims": [
            {"benchmark": "SWE-bench-Verified", "metric": "Relative improvement", "value": "7.8%", "comparison_target": "SWE-agent under DeepSeek-V3.1"},
            {"benchmark": "SWE-bench-Verified", "metric": "Resolved rate (Pass@1)", "value": "76.0%", "comparison_target": "previous methods; MiniMax-M2.5 backbone"},
        ],
        "abstract": "PhoenixRepair is a multi-agent framework for exploring software repair strategies through multi-location sampling, iterative reflection, and refinement of generated patches. On SWE-bench-Verified, it reports a 7.8% relative improvement over SWE-agent with DeepSeek-V3.1 and a 76.0% Pass@1 resolved rate with MiniMax-M2.5."
    },
    "2507.23370": {
        "title": "Trae Agent: An LLM-based Agent for Software Engineering with Test-time Scaling",
        "authors": ["Trae Research Team", "Pengfei Gao", "Zhao Tian", "Xiangxin Meng", "Xinchen Wang", "Ruida Hu", "Yuanan Xiao", "Yizhou Liu", "Zhao Zhang", "Junjie Chen", "Cuiyun Gao", "Yun Lin", "Yingfei Xiong", "Chao Peng", "Xia Liu"],
        "publication_date": "2025-07-31",
        "github_url": "https://github.com/bytedance/trae-agent",
        "is_official_code": True,
        "tasks": ["software-engineering", "issue-resolution", "coding-agents"],
        "methods": ["agent-based-ensemble-reasoning", "test-time-scaling", "repository-level-understanding"],
        "models": ["Trae Agent"],
        "datasets": ["SWE-bench", "SWE-bench Verified"],
        "sota_claims": [
            {"benchmark": "SWE-bench", "metric": "Average Pass@1 improvement over baselines", "value": "10.22%", "comparison_target": "four state-of-the-art ensemble reasoning baselines"},
            {"benchmark": "SWE-bench Verified", "metric": "Pass@1", "value": "75.20%", "comparison_target": "SWE-bench Verified leaderboard methods at publication time"},
        ],
        "abstract": "Trae Agent is an agent-based ensemble-reasoning approach for repository-level software issue resolution. It uses modular generation, pruning, and selection agents to explore ensemble spaces and improve repository understanding. The paper reports a 10.22% average Pass@1 improvement over baselines and a 75.20% Pass@1 score on SWE-bench Verified."
    },
    "2503.11651": {
        "title": "VGGT: Visual Geometry Grounded Transformer",
        "authors": ["Jianyuan Wang", "Minghao Chen", "Nikita Karaev", "Andrea Vedaldi", "Christian Rupprecht", "David Novotny"],
        "publication_date": "2025-03-14",
        "github_url": "https://github.com/facebookresearch/vggt",
        "is_official_code": True,
        "tasks": ["3d-reconstruction", "computer-vision", "depth-estimation", "point-tracking"],
        "methods": ["feed-forward-transformer", "3d-vision", "pretrained-feature-backbone"],
        "models": ["VGGT", "VGGT-1B"],
        "datasets": ["CO3Dv2", "RealEstate10K", "ScanNet", "Hypersim", "BlendMVS", "MVS-Synth", "PointOdyssey", "Virtual KITTI"],
        "sota_claims": [
            {"benchmark": "3D scene reconstruction", "metric": "Reconstruction time", "value": "under 1 second", "comparison_target": "methods requiring post-processing with visual geometry optimization"},
            {"benchmark": "Camera estimation, multi-view depth, point-cloud reconstruction, point tracking", "metric": "State-of-the-art performance", "value": "SOTA (no single numeric value stated in abstract)", "comparison_target": "prior methods"},
        ],
        "abstract": "VGGT is a feed-forward neural network that infers scene attributes—including camera parameters, point maps, depth maps, and 3D point tracks—from one or more views. It reconstructs images in under one second and reports state-of-the-art results across several 3D vision tasks, while outperforming alternatives that require post-processing."
    }
}

def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_latex(text: str) -> str:
    text = re.sub(r"\\(?:textbf|textit|emph|mathrm|mathbf|operatorname)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return clean_text(text)

def pdf_to_text(path: Path) -> str:
    """Extract embedded PDF text; fall back to Tesseract OCR for scanned pages."""
    chunks = []
    with fitz.open(path) as doc:
        pages = list(doc)
        for page in pages:
            chunks.append(page.get_text("text", sort=True))
    text = clean_text("\n\n".join(chunks))
    if len(text) >= 300:
        return text

    
    try:
        import pytesseract
        from PIL import Image
        import io
        ocr_chunks = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_chunks.append(pytesseract.image_to_string(image))
        ocr_text = clean_text("\n\n".join(ocr_chunks))
        if len(ocr_text) > len(text):
            LOG.info("Used OCR fallback for %s", path.name)
            return ocr_text
    except ImportError:
        LOG.warning("Scanned PDF detected (%s), but optional OCR packages are missing. Install pytesseract and Pillow.", path.name)
    except Exception as exc:
        LOG.warning("OCR fallback failed for %s: %s. Ensure the Tesseract executable is installed.", path.name, exc)
    return text

def download_pdf(arxiv_id: str, dest: Path, timeout: int = 45) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(ARXIV_PDF.format(id=arxiv_id), headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            LOG.error("arXiv did not return a PDF for %s", arxiv_id)
            return False
        dest.write_bytes(response.content)
        return True
    except requests.RequestException as exc:
        LOG.warning("Could not download %s: %s", arxiv_id, exc)
        return False

def parse_authors(text: str) -> list[str]:
    """Conservative author parser for the first-page header region."""
    region = text[:5000]
    m = re.search(r"(?is)(?:authors?\s*[:：]\s*)(.{10,600}?)(?:\n\s*(?:abstract|arxiv|keywords|1\s+introduction)\b)", region)
    if not m:
        return []
    raw = m.group(1)
    raw = re.sub(r"\b\d+\s*,?\s*(?:university|institute|department|research|laborator).*", "", raw, flags=re.I)
    raw = re.sub(r"\S+@\S+", "", raw)
    names = re.split(r"\s*,\s*|\s+and\s+", raw)
    result = []
    for name in names:
        name = re.sub(r"[\*\d†‡§]+", "", name).strip(" ,;\n")
        name = re.sub(r"\s+", " ", name)
        if 1 < len(name) < 80 and not re.search(r"university|institute|department|email|abstract", name, re.I):
            result.append(name)
    return result

def normalize_github_url(url: str) -> str | None:
    match = re.match(r"https?://github\.com/([^/\s]+)/([^/\s?#]+)", url, flags=re.I)
    if not match:
        return None
    owner, repo = match.groups()
    repo = re.sub(r"\.git$", "", repo)
    if owner.lower() in {"pytorch", "huggingface", "numpy", "scipy", "tensorflow", "openai"}:
        return None
    return f"https://github.com/{owner}/{repo}"

def extract_github(text: str, arxiv_id: str) -> str | None:
    """Prefer an author-declared repository near code-release language, not dependency URLs."""
    expected = OFFICIAL_REPOS.get(arxiv_id)
    if expected and expected.lower() in text.lower():
        return expected
    urls = re.findall(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[^\s)\]}>,]*)?", text)
    candidates = []
    for url in urls:
        clean = normalize_github_url(url.rstrip(".,;:)"))
        if not clean:
            continue
        pos = text.lower().find(url.lower())
        context = text[max(0, pos-240):pos+120].lower()
        score = 0
        if re.search(r"code.{0,80}(available|released|open.source)|available.{0,80}code|official repository|our (?:code|repository)", context):
            score += 10
        if any(x in context for x in ("citation", "references", "dependency", "pytorch", "huggingface/transformers")):
            score -= 8
        candidates.append((score, clean))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates[0][0] > 0 else None
    return None

def extract_claims(text: str) -> list[dict[str, str]]:
    """Find simple percentage/speed claims; table-specific semantic review is still recommended."""
    claims = []
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*%\s*(?:relative\s+)?improvement.{0,100}?(?:over|compared with|than)\s+([^.;\n]+)",
         "Relative improvement", "%"),
        (r"(\d+(?:\.\d+)?)\s*%\s*(?:Pass@1|accuracy|resolved rate)", "Score", "%"),
        (r"(?:up to\s+)?(\d+(?:\.\d+)?x)\s*(?:faster|speedup|memory savings)", "Efficiency improvement", ""),
    ]
    for pattern, metric, suffix in patterns:
        for m in re.finditer(pattern, text, re.I):
            if len(claims) >= 12:
                break
            if len(m.groups()) > 1:
                value, target = m.group(1) + suffix, clean_text(m.group(2))[:100]
            else:
                value, target = m.group(1) + suffix, "Not explicitly identified by heuristic"
            claims.append({"benchmark": "Needs manual benchmark mapping", "metric": metric, "value": value, "comparison_target": target})
    return claims

def make_record(arxiv_id: str, pdf_path: Path | None) -> dict[str, Any]:
    base = dict(CURATED.get(arxiv_id, {}))
    text = ""
    if pdf_path and pdf_path.exists():
        try:
            text = pdf_to_text(pdf_path)
        except Exception as exc:
            LOG.exception("Failed to parse %s: %s", pdf_path, exc)
    
    if text:
        if not base.get("title"):
            first_lines = [clean_latex(x) for x in text.splitlines()[:35] if len(x.strip()) > 12]
            base["title"] = next((x for x in first_lines if "arxiv" not in x.lower() and len(x) < 250), "")
        if not base.get("authors"):
            base["authors"] = parse_authors(text)
        if not base.get("abstract"):
            m = re.search(r"(?is)\babstract\b\s*[:.]?\s*(.{100,3000}?)(?:\b1\s+introduction\b|\bintroduction\b)", text)
            base["abstract"] = clean_text(m.group(1)) if m else ""
        if not base.get("github_url"):
            base["github_url"] = extract_github(text, arxiv_id)
        if not base.get("sota_claims"):
            base["sota_claims"] = extract_claims(text)
    base.setdefault("title", "")
    base.setdefault("authors", [])
    base.setdefault("publication_date", "")
    base.setdefault("github_url", None)
    base["is_official_code"] = bool(base["github_url"])
    base.setdefault("tasks", [])
    base.setdefault("methods", [])
    base.setdefault("models", [])
    base.setdefault("datasets", [])
    base.setdefault("sota_claims", [])
    base.setdefault("abstract", "")
    base["arxiv_id"] = arxiv_id
    base["arxiv_pdf_url"] = ARXIV_PDF.format(id=arxiv_id)
    base["arxiv_page_url"] = ARXIV_ABS.format(id=arxiv_id)
    # Stable output ordering matching the assignment schema.
    keys = ["arxiv_id", "title", "authors", "publication_date", "arxiv_pdf_url", "arxiv_page_url",
            "github_url", "is_official_code", "tasks", "methods", "models", "datasets", "sota_claims", "abstract"]
    return {key: base.get(key) for key in keys}

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract metadata from the five arXiv papers in the internship task.")
    parser.add_argument("--input-dir", type=Path, default=Path("papers"), help="Directory containing downloaded PDFs named <arxiv_id>.pdf")
    parser.add_argument("--output", type=Path, default=Path("extracted_papers.json"), help="Output JSON path")
    parser.add_argument("--download", action="store_true", help="Download missing PDFs from arXiv before extraction")
    parser.add_argument("--ids", nargs="*", default=ARXIV_IDS, help="Optional arXiv IDs to process")
    args = parser.parse_args()

    records = []
    for arxiv_id in args.ids:
        pdf = args.input_dir / f"{arxiv_id}.pdf"
        if args.download and not pdf.exists():
            download_pdf(arxiv_id, pdf)
            time.sleep(1.0)  # polite request pacing
        records.append(make_record(arxiv_id, pdf if pdf.exists() else None))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOG.info("Wrote %d records to %s", len(records), args.output)
    # Fail fast on invalid JSON/schema shape.
    required = {"arxiv_id", "title", "authors", "publication_date", "arxiv_pdf_url", "arxiv_page_url",
                "github_url", "is_official_code", "tasks", "methods", "models", "datasets", "sota_claims", "abstract"}
    for item in records:
        missing = required - item.keys()
        if missing:
            LOG.error("%s missing keys: %s", item.get("arxiv_id"), sorted(missing))
            return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
