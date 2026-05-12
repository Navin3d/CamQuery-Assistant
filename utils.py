import base64
from pathlib import Path


def image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    with path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
