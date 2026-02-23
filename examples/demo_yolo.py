import argparse

import cv2

from faceswitch import YoloDetector, YoloDetectorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO face detection demo.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument(
        "--model",
        default="yolov8n-face.pt",
        help="YOLO model name or local weights path",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Inference device",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0 - 1.0)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold (0.0 - 1.0)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        raise ValueError(f"--conf must be between 0.0 and 1.0, got {args.conf}")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError(f"--iou must be between 0.0 and 1.0, got {args.iou}")

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image: {args.image_path}")

    config = YoloDetectorConfig(
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        model=args.model,
        device=args.device,
    )
    detector = YoloDetector(config=config)
    faces = detector.detect(image)

    for face in faces:
        cv2.rectangle(image, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
        if face.confidence is not None:
            cv2.putText(
                image,
                f"{face.confidence:.2f}",
                (face.x1, max(face.y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    print(f"Total faces detected: {len(faces)}")
    cv2.imshow("YOLO Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()