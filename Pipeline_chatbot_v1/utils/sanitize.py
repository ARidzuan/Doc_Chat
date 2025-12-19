import json
from langchain.schema import Document

ALLOWED_META_TYPES = (str, int, float, bool, type(None))

def sanitize_documents(docs):
    """
    Sanitize metadata for a sequence of document-like objects and return new Document instances
    with cleaned metadata.
    This function iterates over the supplied documents and normalizes each metadata value so it
    can be safely serialized, indexed, or displayed. The original document objects are not
    modified; instead, new Document instances are created with the same page_content and a
    cleaned metadata dict.
    Normalization rules:
    - Values of types in ALLOWED_META_TYPES are left unchanged.
    - list or tuple:
        - If length == 1: replace with the single element.
        - Otherwise: join elements into a single string with ", " as separator.
    - dict or set: JSON-serialize using json.dumps(..., ensure_ascii=False).
    - bytes or bytearray: try to decode with UTF-8 (errors="ignore"); on failure fall back to str(value).
    - Any other types: convert using str(value).
    - If a document has no metadata (metadata is None), an empty metadata dict is used.
    Parameters
    ----------
    docs : Iterable
            An iterable of document-like objects. Each object must have at least:
            - page_content: the document text/content to carry over unchanged.
            - metadata: a mapping (or None) of metadata fields to values.
    Returns
    -------
    list
            A list of newly created Document instances (Document(page_content, metadata=meta))
            whose metadata values have been converted according to the rules above.
    Notes
    -----
    - This function expects ALLOWED_META_TYPES, json, and Document to be defined in the
        surrounding module scope.
    - The function silently handles decoding/serialization errors by falling back to string
        conversions; it does not raise for individual metadata entries.
    - Ordering of metadata keys is preserved as provided by dict(d.metadata or {}).
    Example
    -------
    # Assuming `docs` is a list of objects with .page_content and .metadata attributes:
    cleaned_docs = sanitize_documents(docs)
    """
    cleaned = []
    for d in docs:
        meta = dict(d.metadata or {})
        for k, v in list(meta.items()):
            if isinstance(v, ALLOWED_META_TYPES):
                continue
            
            elif isinstance(v, (list, tuple)):
                if len(v) == 1:
                    meta[k] = v[0]
                else:
                    meta[k] = ", ".join(map(str, v))
            elif isinstance(v, (dict, set)):
                meta[k] = json.dumps(v, ensure_ascii=False)

            elif isinstance(v, (bytes, bytearray)):
                try:
                    meta[k] = v.decode("utf-8", errors="ignore")
                except Exception:
                    meta[k] = str(v)
            else:
                meta[k] = str(v)
        cleaned.append(Document(page_content=d.page_content, metadata=meta))
    return cleaned
