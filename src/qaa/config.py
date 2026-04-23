from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PRIMARY_DATA_FILE = "Database_gg3.xlsx"
SECONDARY_DATA_FILE = "Database_gg.xlsx"
PERIODS_PER_YEAR = 252

SECONDARY_KEEP_COLUMNS = [
    "Bloomberg EuroAgg Government Total Return Index Value Unhedged EUR",
    "Bloomberg US Treasury 0-1 Year Maturity TR Index Unhedged EUR",
    "Bloomberg EM Local Currency Government TR Index Unhedged EUR",
    "MSCI Daily Gross TR Italy EUR",
    "MSCI ACWI Small Cap Price Return EUR Index",
]

DROP_COLUMNS = [
    "S&P GSCI Commodity Total Return - RETURN IND. (OFCL)",
    "MSCI WORLD REAL ESTATE $",
    "NASDAQ COMPOSITE",
]

RENAMED_COLUMNS = [
    "EU Money Mkt",
    "EU Bond",
    "EU Bond Short Term",
    "Global Bond",
    "EM Bond",
    "Global Corp. Bond High Yield",
    "EU Equity",
    "North America Equity",
    "Pacific Equity",
    "EM Equity",
    "World Equity",
    "Global Real Estates",
    "EU Gov. Bonds",
    "US Money Market",
    "EM Foreing Currency Gov. Bond",
    "ITA Equity",
    "Small Cap Equity",
]
