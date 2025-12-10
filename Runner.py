import cv2
from Vision import *
from OCR import *

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lastNumber = None
    previousResultImg = None
    while True:
        ret, frame = cam.read()
        x,y = frame.shape[:2]
        if not ret:
            break
        disp = frame.copy()
        cv2.putText(disp, "Press space to capture, ESC to exit", (0,20), font, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(disp, f"Last number: {lastNumber}", (0,50), font, 0.7, (0,255,0), 2, cv2.LINE_AA)
        if previousResultImg is not None:
            disp = cv2.bitwise_or(disp, previousResultImg)
        cv2.imshow("Camera Display", disp)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == 32:
            print("calculating...")
            img, circles = detect_circles(frame)
            if circles is not None:
                print(len(circles))
                img = clustering(img)
                img = dilate_erode(img, 19)
                previousResultImg = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
                cv2.imshow("Detected number", img)
                number = getNumber(img)
                lastNumber = number
                if number is not None:
                    print("Detected number:", number)
                else:
                    print("No valid number detected.")
            else:
                print("No circles detected, try again.")
    cam.release()
    cv2.destroyAllWindows()