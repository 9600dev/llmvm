import base64
import math
from io import BytesIO

from PIL import Image


def get_closest_aspect_ratio(width: int, height: int) -> tuple:
    # Define allowed aspect ratios (width:height)
    allowed_ratios = [
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
        (1, 6), (1, 7), (1, 8), (1, 9), (2, 3), (2, 4),
        # Transposes
        (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
        (7, 1), (8, 1), (9, 1), (3, 2), (4, 2)
    ]

    current_ratio = width / height
    closest_ratio = min(allowed_ratios,
                       key=lambda r: abs(current_ratio - (r[0]/r[1])))
    return closest_ratio

def calculate_scaled_dimensions(width: int, height: int) -> tuple:
    # Find closest aspect ratio
    ratio_w, ratio_h = get_closest_aspect_ratio(width, height)

    # Scale so at least one side is >= 896px while maintaining aspect ratio
    scale = max(896 / min(width, height), 1.0)
    new_width = min(round(width * scale), 8000)
    new_height = min(round(height * scale), 8000)

    # Adjust to match exact aspect ratio
    if new_width/new_height > ratio_w/ratio_h:
        new_width = round(new_height * (ratio_w/ratio_h))
    else:
        new_height = round(new_width * (ratio_h/ratio_w))

    return (new_width, new_height)

def estimate_token_count(scaled_width: int, scaled_height: int) -> int:
    # Token count estimation based on documentation examples
    pixel_area = scaled_width * scaled_height
    # Rough estimation formula derived from documentation examples
    token_count = int(pixel_area * 0.0016)  # ~1.6 tokens per 1000 pixels
    return token_count

def get_image_token_count(base64_image: str) -> dict:
    try:
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))

        # Get original dimensions
        orig_width, orig_height = image.size

        # Calculate scaled dimensions
        scaled_width, scaled_height = calculate_scaled_dimensions(orig_width,
orig_height)

        # Get closest aspect ratio
        ratio_w, ratio_h = get_closest_aspect_ratio(orig_width, orig_height)

        # Calculate estimated tokens
        tokens = estimate_token_count(scaled_width, scaled_height)

        return {
            "original_dimensions": f"{orig_width}x{orig_height}",
            "scaled_dimensions": f"{scaled_width}x{scaled_height}",
            "aspect_ratio": f"{ratio_w}:{ratio_h}",
            "estimated_tokens": tokens
        }

    except Exception as e:
        return {"error": str(e)}