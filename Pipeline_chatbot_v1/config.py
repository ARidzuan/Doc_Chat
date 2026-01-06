
import os

# Fix OpenMP duplicate library issue (must be set before numpy/scipy/torch imports)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Get the directory where this config file is located (Pipeline_chatbot_v1)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration for local server and document storage
URL = "http://192.168.11.99:8000"   # base URL for any local API or service
DOC_FOLDER = r"D:\AVSimulation\SCANeRstudio_2025\doc"  # directory containing documents to index
CHROMA_PATH = os.path.join(_CONFIG_DIR, "chroma_db_image1")  # local path/name for Chroma DB storage

ENABLE_SEGMENTATION = True  # Set to False to use single collection (legacy mode)
CHROMA_BASE_PATH = os.path.join(_CONFIG_DIR, "chroma_db_segmented")  # Base directory for segmented collections

# Document categories based on folder structure
DOC_CATEGORIES = [
    "ANALYSIS",      # Analysis tools and workflows
    "COMPUTE",       # Compute modules
    "MODELS",        # 3D models and assets
    "SIMULATION",    # Simulation configuration
    "STUDIO",        # Studio general
    "Studio_APIs",   # API reference documentation
    "TERRAIN",       # Terrain creation tools
    "UNREAL",        # Unreal Engine integration
    "VEHICLE",       # Vehicle dynamics and configuration
]

# Fallback collection for root-level docs and uncategorized content
DEFAULT_COLLECTION = "general"


# Models used for embeddings, reranking and the LLM
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # embedding model for general docs
TOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # smaller topic-level embeddings
RERANKER_MODEL = "BAAI/bge-reranker-large"  # cross-encoder for reranking retrieved passages
LLM_Model = "mistral"            # identifier for the large language model to use

# CUDA toggle (reads from environment variable USE_CUDA).
# Note: bool(os.environ.get("USE_CUDA")) will be True for any non-empty string.
USE_CUDA = bool(os.environ.get("USE_CUDA"))

# Chunking parameters for document splitting
CHUNK_SIZE = 800          # target chunk size (characters/tokens depending on splitter)
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# Maximum Marginal Relevance (MMR) parameters for retrieval diversification
MMR_K = 20                # number of items to select with MMR
MMR_FETCH_K = 60          # number of items to initially fetch before applying MMR
MMR_LAMBDA = 0.9          # trade-off between relevance and diversity (0..1)

# LLM generation settings
LLM_TEMPERATURE = 0.3     # balanced temperature for natural yet accurate responses (increased from 0.1)

# Supported document formats
Text_format = {".txt", ".md", ".docx"}  # plaintext-like formats
Html_format = {".html", ".htm"}         # HTML formats
Pdf_format = ".pdf"                     # PDF format (single string kept for compatibility)
Image_format = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg", ".tif", ".tiff"}  # image extensions

# --- Memory / conversation history settings ---
MEMORY_PATH = "data/memory.json"    # path to persist conversation memory
MEMORY_WINDOW_TURNS = 6             # how many recent turns to keep in short-term memory
MEMORY_SIM_THRESHOLD = 0.72         # similarity threshold for considering memory items relevant
MEMORY_MIN_TOPIC_LEN = 2            # minimum length for a topic to be stored
MEMORY_RECALL_TOP_K = 3             # how many memory items to recall for context

# Clarification behavior (when system asks follow-up questions)
ENABLE_CLARIFICATION = True
CLARIFY_MAX_TURNS = 2   # max number of clarification turns allowed
CLARIFY_MIN_DOCS = 1    # min number of docs needed to trigger clarification logic (reduced from 2)
CLARIFY_MIN_SCORE = 0.25  # minimum relevance score to consider a doc for clarification (reduced from 0.35)

# --- LLM Usage Tracking Settings ---
# Local tracking (always enabled, unlimited)
LOCAL_TRACKING_DB = "data/llm_tracking.db"  # SQLite database for local metrics

