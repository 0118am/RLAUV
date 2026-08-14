"""USD assets used by Isaac Sim."""

from pathlib import Path


ASSET_DIR = Path(__file__).resolve().parent
T60_USD_PATH = ASSET_DIR / "t60_auv.usd"

__all__ = ["ASSET_DIR", "T60_USD_PATH"]
