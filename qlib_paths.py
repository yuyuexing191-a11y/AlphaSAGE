import os


CN_QLIB_PATH = "/home/xyy/.qlib/qlib_data/cn_data"
US_QLIB_PATH = "/your_path/data/qlib_data/us_data_qlib"


def get_qlib_path(instrument: str) -> str:
    """Return the Qlib provider path for the requested market."""
    if instrument == "sp500":
        return os.environ.get("US_QLIB_PATH", US_QLIB_PATH)
    return os.environ.get("CN_QLIB_PATH", CN_QLIB_PATH)
