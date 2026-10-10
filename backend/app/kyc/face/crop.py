from PIL import Image
from typing import Tuple
from backend.app.kyc.face.detect import FaceBox

def canonical_face_crop(
    image: Image.Image,
    face_box: FaceBox,
    margin_ratio: float = 1.25,
    target_size: Tuple[int, int] = (224, 224),
    resample: Image.Resampling = Image.Resampling.BILINEAR
) -> Image.Image:
    img_w, img_h = image.size
    cx = face_box.x + face_box.w / 2.0
    cy = face_box.y + face_box.h / 2.0

    side = max(face_box.w, face_box.h) * margin_ratio

    half_side = side / 2.0
    left = int(round(cx - half_side))
    top = int(round(cy - half_side))
    right = int(round(cx + half_side))
    bottom = int(round(cy + half_side))

    crop_l = max(0, left)
    crop_t = max(0, top)
    crop_r = min(img_w, right)
    crop_b = min(img_h, bottom)

    cropped = image.crop((crop_l, crop_t, crop_r, crop_b))
    resized = cropped.resize(target_size, resample=resample)
    return resized
