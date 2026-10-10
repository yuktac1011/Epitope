import io
import hashlib
from typing import Tuple, Dict, Any, Union
from PIL import Image, ImageOps

class InputValidationError(ValueError):
    """Raised when an uploaded file violates safety or format limits."""
    pass

def safe_load_image(
    input_data: Union[bytes, str],
    max_file_bytes: int = 10 * 1024 * 1024,
    max_image_pixels: int = 50_000_000,
    max_dimension: int = 4096,
    allowed_formats: tuple = ("JPEG", "PNG", "WEBP")
) -> Tuple[Image.Image, Dict[str, Any]]:
    Image.MAX_IMAGE_PIXELS = max_image_pixels

    if isinstance(input_data, str):
        with open(input_data, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(input_data, bytes):
        raw_bytes = input_data
    else:
        raise InputValidationError("Input must be file path or bytes.")

    file_size = len(raw_bytes)
    if file_size == 0:
        raise InputValidationError("Empty input file (0 bytes).")
    if file_size > max_file_bytes:
        raise InputValidationError(f"File size {file_size} exceeds limit of {max_file_bytes} bytes.")

    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
    except Exception as e:
        raise InputValidationError(f"Could not decode image format: {e}")

    detected_format = pil_img.format
    if not detected_format or detected_format.upper() not in allowed_formats:
        raise InputValidationError(
            f"Unsupported format '{detected_format}'. Allowed formats: {allowed_formats}"
        )

    is_animated = getattr(pil_img, "is_animated", False)
    if is_animated:
        pil_img.seek(0)

    had_exif = False
    try:
        exif = pil_img.getexif()
        if exif and len(exif) > 0:
            had_exif = True
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    w, h = pil_img.size
    if w > max_dimension or h > max_dimension:
        raise InputValidationError(
            f"Dimensions {w}x{h} exceed maximum allowed dimension {max_dimension}."
        )

    had_alpha = False
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        had_alpha = True
        background = Image.new("RGB", pil_img.size, (255, 255, 255))
        converted = pil_img.convert("RGBA")
        background.paste(converted, mask=converted.split()[3])
        pil_img = background
    elif pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    meta = {
        "sha256": sha256_hash,
        "format": detected_format,
        "width": w,
        "height": h,
        "had_exif": had_exif,
        "had_alpha": had_alpha,
        "animated": is_animated
    }

    return pil_img, meta
