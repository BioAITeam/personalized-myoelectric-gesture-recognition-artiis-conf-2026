from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import pypdfium2 as pdfium


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LATEX_DIR = ROOT.parent / "paper"
DEFAULT_OUTPUT_DIR = ROOT / ".tmp" / "paper_review"


def render_pdf(pdf_path: Path, output_dir: Path, prefix: str, scale: float) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    paths: list[Path] = []
    for index, page in enumerate(document, start=1):
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        output_path = output_dir / f"{prefix}_page_{index:02d}.png"
        image.save(output_path)
        paths.append(output_path)
    return paths


def contact_sheet(images: list[Path], output_path: Path, thumb_width: int = 420) -> None:
    thumbs: list[Image.Image] = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        ratio = thumb_width / image.width
        thumb = image.resize((thumb_width, max(1, int(image.height * ratio))))
        thumb = ImageOps.expand(thumb, border=8, fill="white")
        thumbs.append(thumb)

    if not thumbs:
        return

    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    cell_width = max(thumb.width for thumb in thumbs)
    cell_height = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex-dir", type=Path, default=DEFAULT_LATEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=float, default=2.0)
    args = parser.parse_args()

    paper_pdf = args.latex_dir / "paper.pdf"
    paper_dir = args.output_dir / "paper_pdf"
    figure_dir = args.output_dir / "figure_pdfs"

    paper_pages = render_pdf(paper_pdf, paper_dir, "paper", args.scale)
    contact_sheet(paper_pages, args.output_dir / "paper_contact_sheet.png")

    figure_pages: list[Path] = []
    for figure_pdf in sorted((args.latex_dir / "figs").glob("*.pdf")):
        figure_pages.extend(render_pdf(figure_pdf, figure_dir, figure_pdf.stem, args.scale))
    contact_sheet(figure_pages, args.output_dir / "figure_pdf_contact_sheet.png")

    print(f"Paper pages: {len(paper_pages)}")
    print(f"Figure PDF pages: {len(figure_pages)}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
