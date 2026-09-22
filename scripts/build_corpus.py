"""Build the e-commerce support document corpus used by the RAG assistant.

The corpus is derived from the public Bitext customer-support KB
(https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset):
27,000+ instruction/response pairs grouped by fine-grained support intent. Each
intent group becomes one FAQ-style document in ``data/documents/``, and a couple
of documents are also exported as real PDFs so the extraction pipeline has
something other than plain text to parse.

Usage:
    python scripts/build_corpus.py            # builds data/documents/ only
    python scripts/build_corpus.py --all      # also keeps the full raw CSV at data/raw/
"""
import argparse
import re
import shutil
import textwrap
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "documents"
RAW_DIR = DATA_DIR / "raw"

DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
RAW_CSV = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
MAX_PAIRS_PER_DOC = 200  # corpus stays small enough to commit to git

PDF_INTENTS = ["check_refund_policy", "cancel_order"]

TITLE_OVERRIDES = {
    "contact_customer_service": "Contacting Customer Service",
    "contact_human_agent": "Reaching a Human Agent",
    "check_cancellation_fee": "Cancellation Fees",
}


def clean(text: str) -> str:
    text = text.replace("{{", "(").replace("}}", ")")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_case(intent: str) -> str:
    return TITLE_OVERRIDES.get(intent, intent.replace("_", " ").title())


def to_markdown(intent: str, rows) -> str:
    lines = [
        f"# {title_case(intent)}",
        "",
        f"> Support knowledge base topic: ``{intent}``. "
        "Answers below are the authoritative responses for this topic.",
        "",
    ]
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        lines.append(f"## Q{i}: {clean(row['instruction'])}")
        lines.append("")
        lines.append(clean(row["response"]))
        lines.append("")
    return "\n".join(lines)


def build_docs(df) -> None:
    if DOCS_DIR.exists():
        shutil.rmtree(DOCS_DIR)
    DOCS_DIR.mkdir(parents=True)
    intents = df["intent"].unique()
    docs = []
    for idx, intent in enumerate(sorted(intents)):
        rows = df[df["intent"] == intent].head(MAX_PAIRS_PER_DOC)
        docs.append((intent, rows))
        (DOCS_DIR / f"{idx + 1:02d}_{intent}.md").write_text(
            to_markdown(intent, rows), encoding="utf-8"
        )
        print(f"wrote docs/{idx + 1:02d}_{intent}.md ({len(rows)} pairs)")
    return docs


def build_pdfs(docs) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], leading=14, spaceAfter=6
    )
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]

    for intent, rows in docs:
        if intent not in PDF_INTENTS:
            continue
        dest = DOCS_DIR / f"{intent}.pdf"
        doc = SimpleDocTemplate(str(dest), pagesize=letter,
                                leftMargin=1 * inch, rightMargin=1 * inch)
        story = [Paragraph(title_case(intent), h1)]
        for i, (_, row) in enumerate(rows.head(30).iterrows(), 1):
            story.append(Paragraph(f"Q{i}. {clean(row['instruction'])}", h2))
            story.append(Paragraph(clean(row["response"]), body))
            story.append(Spacer(1, 4))
        doc.build(story)
        print(f"wrote docs/{intent}.pdf (first 30 Q&A)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="save the full 27k-row raw CSV too")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET)
    df = ds["train"].to_pandas()[["instruction", "category", "intent", "response"]]
    df["instruction"] = df["instruction"].map(clean)
    df["response"] = df["response"].map(clean)
    print(f"Loaded {len(df):,} support pairs from Bitext")

    docs = build_docs(df)
    build_pdfs(docs)

    if args.all:
        df.to_csv(RAW_DIR / RAW_CSV, index=False)
        print(f"saved full raw CSV ({len(df):,} rows) to data/raw/{RAW_CSV}")


if __name__ == "__main__":
    main()