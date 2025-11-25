import cv2

webcam = cv2.VideoCapture(0)

def video_capture():
    while True:
        ret, frame = webcam.read()

        if ret == True:
            cv2.imshow('Cam', frame)

            key = cv2.waitKey(1)

            if key == ord('q'):
                break
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_capture()