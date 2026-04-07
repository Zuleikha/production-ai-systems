"""
Object detection inference engine using Faster R-CNN ResNet-50 FPN.
Supports single-image and batch processing with configurable confidence and NMS thresholds.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.ops
import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from visualization import DetectionVisualizer, VideoVisualizer

logger = logging.getLogger(__name__)


class ModelLoader:
    """Load Faster R-CNN models from checkpoint or fall back to pretrained weights."""

    def load_model(
        self, model_path: str, device: torch.device, num_classes: int = 91
    ) -> torch.nn.Module:
        """
        Load a model checkpoint; fall back to COCO pretrained weights if not found.

        Args:
            model_path: Path to saved checkpoint (.pth).
            device: Target device for inference.
            num_classes: Number of output classes.

        Returns:
            Model in eval mode on the target device.
        """
        if Path(model_path).exists():
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
                state_dict = (
                    checkpoint["model_state_dict"]
                    if "model_state_dict" in checkpoint
                    else checkpoint
                )
                model.load_state_dict(state_dict)
                logger.info(f"Checkpoint loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e} — falling back to pretrained")
                model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            logger.info(f"No checkpoint at {model_path} — loading pretrained COCO weights")
            model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        model.to(device).eval()
        return model


@dataclass
class DetectionResult:
    """Structured container for model predictions."""
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    processing_time: float
    image_shape: Tuple[int, int]


class ObjectDetector:
    """Faster R-CNN inference with configurable confidence/NMS thresholds."""

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        device: str = "auto",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ):
        """
        Args:
            model_path: Path to model checkpoint.
            class_names: Class name list matching model output indices.
            device: 'auto', 'cpu', or 'cuda'.
            confidence_threshold: Minimum score to keep a detection.
            nms_threshold: IoU threshold for Non-Maximum Suppression.
        """
        logging.basicConfig(level=logging.INFO)

        self.model_path = model_path
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.inference_times: List[float] = []

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        # Build transform once — reused on every inference call.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        self.model = ModelLoader().load_model(model_path, self.device, len(class_names))
        logger.info(f"Detector ready on {self.device}")

        self.visualizer = DetectionVisualizer(class_names, confidence_threshold)
        self.video_visualizer = VideoVisualizer(self.visualizer)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert a BGR numpy image to a normalised tensor batch."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb)
        return tensor.unsqueeze(0).to(self.device)

    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Apply torchvision NMS and return surviving indices."""
        keep = torchvision.ops.nms(
            torch.from_numpy(boxes),
            torch.from_numpy(scores),
            self.nms_threshold,
        )
        return keep.cpu().numpy()

    def postprocess_detections(
        self,
        outputs: List[Dict],
        original_shape: Tuple[int, int],
    ) -> DetectionResult:
        """Filter by confidence and apply NMS to raw model outputs."""
        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        keep = scores >= self.confidence_threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        if len(boxes) > 0:
            keep_indices = self.apply_nms(boxes, scores)
            boxes, scores, labels = boxes[keep_indices], scores[keep_indices], labels[keep_indices]

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=labels,
            processing_time=0.0,  # Set by caller after timing.
            image_shape=original_shape,
        )

    def detect_single_image(self, image: np.ndarray) -> DetectionResult:
        """Run end-to-end detection on a single BGR image."""
        start = time.time()

        tensor = self.preprocess_image(image)
        with torch.no_grad():
            outputs = self.model(tensor)

        result = self.postprocess_detections(outputs, image.shape[:2])
        result.processing_time = time.time() - start
        self.inference_times.append(result.processing_time)

        return result


def main():
    """Smoke-test the inference engine with a synthetic image."""
    MODEL_PATH = "models/best_model.pth"
    CLASS_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    detector = ObjectDetector(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES,
        confidence_threshold=0.5,
        nms_threshold=0.4,
    )

    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect_single_image(test_image)
    print(f"Detected {len(result.boxes)} objects in {result.processing_time:.3f}s")

    if detector.inference_times:
        avg = np.mean(detector.inference_times)
        print(f"Average inference: {avg:.3f}s  ({1.0 / avg:.1f} FPS)")

    print(f"Device: {detector.device}")


if __name__ == "__main__":
    main()
