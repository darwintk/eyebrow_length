#coding=utf-8
# faces detect
import cv2 as cv
import dlib

path = "./ori_img/chelsea1.jpg"
img = cv.imread(path)
cv.imshow("image", img)
cv.waitKey(0)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
dets = detector(gray,1)
for face in dets:
    # 在图片中标注人脸，并显示
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    cv.rectangle(img, (left, top), (right, bottom), (0 ,0, 255), 2)
    cv.imshow("image", img)
    cv.waitKey(0)
cv.imshow("image", img)
cv.waitKey(0)
cv.imwrite("./out_img/chelsea1_faces.jpg",img)
cv.destroyAllWindows()