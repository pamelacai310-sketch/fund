from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SUPPORTED_TABLE_EXTS = {".xlsx", ".xls", ".csv"}

DOC_KEYWORDS = [
    "factsheet",
    "fund factsheet",
    "monthly report",
    "fund report",
    "product key facts",
    "KFS",
    "基金月报",
    "产品资料概要",
    "基金资料",
]

SOCIAL_PLATFORMS = ["小红书", "微博", "抖音"]
SOCIAL_RECENT_DAYS = 183
