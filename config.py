import os
from dotenv import load_dotenv

load_dotenv()

# Airtable Configuration
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "Bitcoin_Market_Data")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "bitcoin-strategies"

# Backtest Configuration
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_COMMISSION = 0.001  # 0.1%
DEFAULT_SLIPPAGE = 0.0005  # 0.05%
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = "2024-12-31"

# Output Configuration
REPORTS_DIR = "outputs/reports"
CHARTS_DIR = "outputs/charts"
DATA_CACHE_DIR = "data/raw"