import cv2
class CommonFunc():
    def showPic(img):
        cv2.imshow("Study", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

