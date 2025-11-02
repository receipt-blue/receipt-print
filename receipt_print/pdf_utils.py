import sys
from typing import Iterable, List, Optional, Tuple, Union

from pdf2image import convert_from_bytes
from PIL import Image

DEFAULT_PDF_DPI = 300


def filter_pages(
    pages: List[Image.Image], page_filter: Optional[Tuple]
) -> List[Image.Image]:
    """Filter pages based on page selection criteria"""
    if not page_filter:
        return pages

    filter_type = page_filter[0]
    total_pages = len(pages)

    if filter_type == "range":
        _, start, end = page_filter
        start_idx = start - 1  # handle 1-indexing
        end_idx = end - 1 if end != -1 else total_pages - 1
        start_idx = max(0, min(start_idx, total_pages - 1))
        end_idx = max(start_idx, min(end_idx, total_pages - 1))

        return pages[start_idx : end_idx + 1]

    elif filter_type == "pages":
        _, page_nums = page_filter
        indices = [p - 1 for p in page_nums if 1 <= p <= total_pages]
        return [pages[i] for i in indices]

    return pages


def pdf_to_images(
    pdf_paths: Iterable[str], page_filter: Optional[Tuple] = None
) -> Tuple[List[Image.Image], List[str]]:
    """Convert PDF pages to images, return (images, names)"""
    images: List[Image.Image] = []
    names: List[str] = []

    for path in pdf_paths:
        try:
            with open(path, "rb") as fh:
                pages = convert_from_bytes(fh.read(), dpi=DEFAULT_PDF_DPI)

            filtered_pages = filter_pages(pages, page_filter)

            if filtered_pages:
                if page_filter and page_filter[0] == "range":
                    start_page = page_filter[1]
                    page_nums = list(
                        range(start_page, start_page + len(filtered_pages))
                    )
                elif page_filter and page_filter[0] == "pages":
                    # only include page numbers that exist in the PDF
                    total_pages = len(pages)
                    page_nums = [p for p in page_filter[1] if 1 <= p <= total_pages]
                else:
                    page_nums = list(range(1, len(filtered_pages) + 1))

                images.extend(filtered_pages)
                names.extend([f"{path}#page{num}" for num in page_nums])
        except Exception as e:
            sys.stderr.write(f"Warning: could not open {path}: {e}\n")

    return images, names


def pdf_to_text(paths: Iterable[str], page_filter: Optional[Tuple] = None) -> str:
    from markitdown import MarkItDown

    # for text conversion, convert the whole PDF and then extract pages
    md = MarkItDown(enable_plugins=False)
    chunks: List[str] = []

    for p in paths:
        try:
            res = md.convert(p)
            text_content = res.text_content

            if page_filter:
                sys.stderr.write(
                    f"Warning: Page filtering for text format is not fully supported. "
                    f"Converting entire PDF {p}.\n"
                )

            chunks.append(text_content)
        except Exception as e:
            sys.stderr.write(f"Error converting {p}: {e}\n")
    return "\n\n".join(chunks)
