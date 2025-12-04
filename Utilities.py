import cv2

webcam = cv2.VideoCapture(0)


def video_capture(on_space_callback=None):
    print("Press SPACE to analyze sign")
    print("Press Q to quit")

    while True:
        ret, frame = webcam.read()

        if ret == True:
            cv2.imshow('Cam', frame)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break
            elif key == ord(' ') and on_space_callback:
                on_space_callback(frame)

if __name__ == "__main__":
    video_capture()