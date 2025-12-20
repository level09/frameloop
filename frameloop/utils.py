"""Utility functions."""

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def prepare_image_input(image_path: str):
    """Return URL string or file handle for local files."""
    if is_url(image_path):
        return image_path
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return open(path, "rb")


def generate_output_filename(input_path: str, output_type: str, extension: str) -> str:
    """Generate output filename based on input and type."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "output" if is_url(input_path) else Path(input_path).stem
    return f"{base}_{output_type}_{timestamp}.{extension}"


def save_output(url: str, output_path: str) -> Path:
    """Download and save output file."""
    path = Path(output_path)
    with httpx.stream("GET", url, follow_redirects=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return path


def format_duration(seconds: float) -> str:
    """Format duration in human readable form."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"
