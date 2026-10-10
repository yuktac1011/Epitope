import os
import torch
from torchvision import transforms
from PIL import Image
from typing import Optional
from backend.app.kyc.signals.base import SignalResult, Sample
from backend.app.kyc.model import get_kyc_model
from backend.app.kyc.calibration import TemperatureScaler

class ArtifactSignal:
    name: str = "artifact"

    def __init__(
        self,
        checkpoint_path: Optional[str] = "backend/app/kyc/checkpoints/best_kyc_model.pth",
        device: str = "cpu",
        temperature: float = 1.0
    ):
        self.device = device
        self.model = get_kyc_model(pretrained=False)
        self.model.eval()
        self.model.to(self.device)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.version = "untrained-base"
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.version = os.path.basename(checkpoint_path)

        self.scaler = TemperatureScaler(temperature=temperature)

    def score(self, sample: Sample) -> SignalResult:
        if not sample.frames:
            return SignalResult(
                name=self.name,
                available=False,
                score=None,
                reasons=["NO_FRAMES_PROVIDED"]
            )

        frame = sample.frames[0]
        if isinstance(frame, str):
            image = Image.open(frame).convert("RGB")
        elif isinstance(frame, Image.Image):
            image = frame.convert("RGB")
        else:
            image = Image.fromarray(frame).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            # Calibrate logits with temperature
            probs = self.scaler.calibrate_probs(logits)
            # Class 0 = fake, Class 1 = real
            fake_prob = float(probs[0][0].item())

        reasons = []
        if fake_prob >= 0.75:
            reasons.append("HIGH_SYNTHETIC_PROBABILITY")
        elif fake_prob >= 0.15:
            reasons.append("AMBIGUOUS_FACIAL_TEXTURE")

        return SignalResult(
            name=self.name,
            available=True,
            score=fake_prob,
            confidence=abs(fake_prob - 0.5) * 2.0, # distance from 50% boundary
            reasons=reasons,
            meta={"model_version": self.version}
        )
