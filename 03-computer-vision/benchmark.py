"""
Inference benchmarking script for the object detection pipeline.
Run from the project root: python benchmark.py

Creates test_images/ with synthetic images if the folder does not exist,
then reports latency and confidence statistics to the console.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.inference.detector import ObjectDetector

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"

COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def create_synthetic_images(folder: Path, count: int = 3) -> None:
    """Generate synthetic BGR JPEG test images using numpy (no internet required)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(count):
        img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / f"synthetic_{i}.jpg"), img)
    print(f"Created {count} synthetic test images in {folder}/")


def run_benchmark() -> None:
    if not TEST_IMAGES_DIR.exists() or not any(TEST_IMAGES_DIR.iterdir()):
        create_synthetic_images(TEST_IMAGES_DIR)

    image_paths = sorted(TEST_IMAGES_DIR.glob("*.jpg")) + sorted(TEST_IMAGES_DIR.glob("*.png"))
    if not image_paths:
        print("No images found in test_images/. Aborting.")
        return

    detector = ObjectDetector(
        model_path="models/best_model.pth",
        class_names=COCO_CLASSES,
        confidence_threshold=0.5,
        nms_threshold=0.2,
    )

    latencies_ms: list[float] = []
    all_scores: list[float] = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue
        result = detector.detect_single_image(image)
        latencies_ms.append(result.processing_time * 1000)
        all_scores.extend(result.scores.tolist())

    if not latencies_ms:
        print("No images processed.")
        return

    print("\n--- Benchmark Results ---")
    print(f"Images processed : {len(latencies_ms)}")
    print(f"Average latency  : {np.mean(latencies_ms):.1f} ms")
    print(f"Min latency      : {np.min(latencies_ms):.1f} ms")
    print(f"Max latency      : {np.max(latencies_ms):.1f} ms")
    if all_scores:
        print(f"Avg confidence   : {np.mean(all_scores):.3f}")
    else:
        print("Avg confidence   : N/A (no detections above threshold)")
    print("-------------------------")


if __name__ == "__main__":
    run_benchmark()
