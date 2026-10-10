import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

@dataclass
class FaceDetectionResult:
    n_faces: int
    primary_face: Optional[FaceBox]
    all_faces: List[FaceBox] = field(default_factory=list)
    status: str = "OK"
    reason: Optional[str] = None

class FaceDetector:
    def __init__(
        self,
        model_path: str = "backend/app/kyc/models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.60,
        nms_threshold: float = 0.30,
        dominant_area_ratio: float = 0.60
    ):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.dominant_area_ratio = dominant_area_ratio
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Face detector model not found at: {self.model_path}")
        self._detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (128, 128),
            self.score_threshold,
            self.nms_threshold,
            5000
        )

    def detect(self, image: Image.Image) -> FaceDetectionResult:
        w, h = image.size
        np_img = np.array(image)
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr_img)

        if faces is None or len(faces) == 0:
            return FaceDetectionResult(
                n_faces=0,
                primary_face=None,
                status="NO_FACE",
                reason="NO_FACE"
            )

        detected_boxes: List[FaceBox] = []
        for face_data in faces:
            fx, fy, fw, fh = face_data[:4]
            score = float(face_data[-1])
            landmarks = face_data[4:14].reshape((5, 2))

            bx = max(0, int(fx))
            by = max(0, int(fy))
            bw = min(w - bx, int(fw))
            bh = min(h - by, int(fh))

            if bw > 0 and bh > 0:
                detected_boxes.append(FaceBox(x=bx, y=by, w=bw, h=bh, score=score, landmarks=landmarks))

        if not detected_boxes:
            return FaceDetectionResult(n_faces=0, primary_face=None, status="NO_FACE", reason="NO_FACE")

        detected_boxes.sort(key=lambda b: b.area, reverse=True)
        n_faces = len(detected_boxes)

        if n_faces == 1:
            return FaceDetectionResult(
                n_faces=1,
                primary_face=detected_boxes[0],
                all_faces=detected_boxes,
                status="OK"
            )

        total_area = sum(b.area for b in detected_boxes)
        largest = detected_boxes[0]
        if total_area > 0 and (largest.area / total_area) >= self.dominant_area_ratio:
            return FaceDetectionResult(
                n_faces=n_faces,
                primary_face=largest,
                all_faces=detected_boxes,
                status="OK"
            )

        return FaceDetectionResult(
            n_faces=n_faces,
            primary_face=None,
            all_faces=detected_boxes,
            status="MULTIPLE_FACES",
            reason="MULTIPLE_FACES"
        )
