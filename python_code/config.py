from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT_DIR / "data" / "raw"
PROCESSED_DATA = ROOT_DIR / "data" / "processed"
OUTPUT_GRAPHS = ROOT_DIR / "graphs"

# Create output directories when the configuration is imported.
RAW_DATA.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
OUTPUT_GRAPHS.mkdir(parents=True, exist_ok=True)
