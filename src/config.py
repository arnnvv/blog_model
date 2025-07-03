import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

CORPORATE_SITES_FILE = RAW_DATA_DIR / "ranked_domains.csv"
PERSONAL_SITES_FILE = RAW_DATA_DIR / "searchmysite_urls.csv"

ENRICHED_DATA_FILE = PROCESSED_DATA_DIR / "enriched_data.parquet"
TEMP_JSONL_FILE = PROCESSED_DATA_DIR / \
    "temp_enriched_data.jsonl"

LABEL_MAPPING = {"corporate_seo": 0, "personal_blog": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 8000
NGRAM_RANGE = (1, 2)

LGBM_PARAMS = {
    "objective":        "binary",
    "boosting_type":    "gbdt",
    "learning_rate":    0.08,
    "n_estimators":     400,
    "num_leaves":       21,
    "max_depth": -1,
    "max_bin":          63,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq":     1,
    "min_data_in_leaf": 20,
    "n_jobs":           1,
    "verbosity": -1,
}

MAX_WORKERS = 40
REQUEST_TIMEOUT = 10
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
