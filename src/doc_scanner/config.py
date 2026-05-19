from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# Detection Parameters
RESIZE_WIDTH = 900
CANNY_LOW = 50
CANNY_HIGH = 150
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
