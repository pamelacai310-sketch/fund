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
SUPPORTED_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_PDF_EXTS = {".pdf"}

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
SOCIAL_RECENT_MONTHS = int(os.getenv("SOCIAL_RECENT_MONTHS", "6"))
MAX_DOC_RESULTS_PER_FUND = int(os.getenv("MAX_DOC_RESULTS_PER_FUND", "10"))
MAX_DOC_FETCH_PER_FUND = int(os.getenv("MAX_DOC_FETCH_PER_FUND", "3"))
MAX_SOCIAL_RESULTS_PER_QUERY = int(os.getenv("MAX_SOCIAL_RESULTS_PER_QUERY", "8"))
FETCH_DOCUMENT_TEXT = os.getenv("FETCH_DOCUMENT_TEXT", "1") != "0"

OFFICIAL_DOMAINS = [
    "blackrock.com",
    "jpmorgan.com",
    "allianzgi.com",
    "fidelity.com",
    "schroders.com",
    "ubs.com",
    "hsbc.com",
    "hsbc.com.hk",
    "assetmanagement.hsbc.nl",
    "invesco.com",
    "pimco.com",
    "vanguard.com",
    "franklintempleton.com",
    "valuepartners-group.com",
    "amundi.com",
    "amundi.com.hk",
    "pictet.com",
    "bea-union-investment.com",
    "bea-union-investment.com.hk",
    "sfc.hk",
    "abrdn.com",
    "morganstanley.com",
]
