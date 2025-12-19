import os
from typing import List
from bs4 import BeautifulSoup
from langchain.schema import Document

def extract_image_captions(html_path: str) -> List[Document]:
    """
    Extract image captions and related metadata from a local HTML file.
    This function parses the provided HTML file and attempts to build a short,
    structured caption for each <img> element whose "src" resolves to an
    existing local file. The caption is assembled from available attributes and
    context in a prioritized manner (alt, title, figcaption, parent text, filename).
    For each successfully processed image a Document object is created containing
    the composed caption as page_content and a metadata dictionary describing the
    image and how the caption was derived.
    Parameters
    ----------
    html_path : str
        Path to the local HTML file to parse. The function reads this file using
        UTF-8 encoding (errors ignored) and resolves image "src" attributes
        relative to the directory containing this file.
    Returns
    -------
    List[Document]
        A list of Document objects, one per image that was found and whose
        resolved image file exists on disk. Each Document includes:
          - page_content: str
              A caption string composed of one or more parts separated by ' | '.
              Parts are labeled to indicate their origin, for example:
                'alt: <alt text>', 'title: <title text>',
                'figcaption: <figcaption text>', 'context: <parent text>',
                'filename: <basename>'
          - metadata: dict
              Keys provided:
                - 'source': basename of the image file
                - 'image_path': normalized path to the image file
                - 'modality': 'image_caption'
                - 'html_source': original html_path provided to the function
    Behavior and composition rules
    ------------------------------
    - The HTML is parsed with BeautifulSoup using the 'html.parser'.
    - For each <img> tag:
      1. Skip if no src attribute.
      2. Resolve src relative to the directory of html_path using os.path.join
         and os.path.normpath.
      3. Skip the image if the resolved file does not exist on disk.
      4. Collect caption parts in this order (adding each non-empty part):
         - alt attribute -> added as "alt: <alt text>" (whitespace collapsed/stripped)
         - title attribute -> added as "title: <title text>"
         - figcaption (if the image is inside a <figure>) -> "figcaption: <text>"
         - If none of the above were present, the parent element's text (collapsed
           whitespace) up to 300 characters -> "context: <text>"
         - If still empty, fall back to the image filename -> "filename: <basename>"
      5. Join collected parts with " | " to form Document.page_content.
    - Any exception during parsing/processing is caught; a message is printed
      to stdout in the form:
        "[HTML image parse] {html_path}: {exception}"
    Notes
    -----
    - This function expects a Document class/constructor to be available in the
      calling context. It does not create or validate any external embeddings.
    - Parent element text is truncated to 300 characters to avoid overly long
      caption strings.
    - Attribute and figcaption text are stripped and internal whitespace is
      normalized to single spaces.
    - Only images that resolve to existing local files are returned.
    - The function intentionally swallows exceptions and prints a short message;
      callers who need richer error handling should wrap calls accordingly.
    Example
    -------
    >>> docs = extract_image_captions('/path/to/page.html')
    >>> for doc in docs:
    ...     print(doc.page_content, doc.metadata['image_path'])
    'alt: Example image | figcaption: Example caption' /path/to/images/example.png
    """
    docs = []
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()
        soup = BeautifulSoup(html, "html.parser")
        base_dir = os.path.dirname(html_path)

        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            img_path = os.path.normpath(os.path.join(base_dir, src))
            if not os.path.exists(img_path):
                continue

            parts = []
            alt = (img.get("alt") or "").strip()
            title = (img.get("title") or "").strip()
            if alt:
                parts.append(f"alt: {alt}")
            if title:
                parts.append(f"title: {title}")

            fig = img.find_parent("figure")
            if fig:
                fc = fig.find("figcaption")
                if fc:
                    cap = " ".join(fc.get_text(" ", strip=True).split())
                    if cap:
                        parts.append(f"figcaption: {cap}")

            if not parts:
                parent_text = " ".join((img.parent.get_text(" ", strip=True) if img.parent else "").split())
                if parent_text:
                    parts.append(f"context: {parent_text[:300]}")

            if not parts:
                parts.append(f"filename: {os.path.basename(img_path)}")

            caption_text = " | ".join(parts)
            docs.append(Document(
                page_content=caption_text,
                metadata={
                    "source": os.path.basename(img_path),
                    "image_path": img_path,
                    "modality": "image_caption",
                    "html_source": html_path
                }
            ))
    except Exception as e:
        print(f"[HTML image parse] {html_path}: {e}")
    return docs
