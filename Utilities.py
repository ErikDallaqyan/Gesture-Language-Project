import cv2
from transformers import pipeline
from PIL import Image

pipe = pipeline("image-classification", model="prithivMLmods/Alphabet-Sign-Language-Detection", use_fast=True)


def analyze_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    result = pipe(pil_image)

    print("\n=== Sign Detection ===")
    for pred in result:
        print(f"{pred['label']}: {pred['score']:.2%}")
    print("===================\n")


def video_capture():
    webcam = cv2.VideoCapture(0)

    print("Press SPACE to analyze sign")
    print("Press Q to quit")

    while True:
        ret, frame = webcam.read()

        if ret:
            cv2.imshow('Sign Language Detection', frame)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord(' '):
                analyze_frame(frame)

    webcam.release()
    cv2.destroyAllWindows()

