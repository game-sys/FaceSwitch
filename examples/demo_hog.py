import argparse

import cv2

from faceswitch import HogDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HOG face detection demo.")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image: {args.image_path}")

    detector = HogDetector()
    faces = detector.detect(image)

    for face in faces:
        cv2.rectangle(image, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)

    print(f"Total faces detected: {len(faces)}")
    cv2.imshow("HOG Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
