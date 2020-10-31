# Dlib面部识别以及68个特征点

> 在处理图像上，opencv比较常用，因此除了人脸识别使用了dlib库，其他地方都使用了opencv来进行处理

本代码参考[vipstone/faceai](https://github.com/vipstone/faceai)

## Dlib面部识别
使用dlib中的get_frontal_face_detector()来查找图片中的人脸，代码如下：

``` python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	# 转灰度图
detector = dlib.get_frontal_face_detector()	# 人脸分类器
dets = detector(gray, 1)	# 使用人脸分类器找到图中的人脸
for face in dets:
    # 找到人脸矩形框的四个角点
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    # 在图片中标注人脸，并显示
    cv2.rectangle(img, (left, top), (right, bottom), (255 ,0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
```

完整代码：[facesDetection](./facesDetection.py)

识别的结果如下图所示：
![Chelsea](./out_img/chelsea1_face.jpg)


## Dlib面部识别的68个特征点
除了识别图像中的人脸外，dlib还能找到人脸上的68个特征点
- 面部轮廓 1~17
- 眉毛
   - 左眉毛 18~22
   - 右眉毛 23~27
- 鼻梁 28~31
- 鼻子下沿 32~36
- 眼睛
   - 左眼 37~42
   - 右眼 43~48
- 嘴巴外轮廓49~60
- 嘴巴内轮廓 61~68 

如图所示：
![Ronald](./out_img/Ronald1_68points.jpg)

代码如下：

``` python
# 获取人脸检测器
predictor = dlib.shape_predictor(
    "./shape_predictor_68_face_landmarks.dat")

dets = detector(gray, 1)
for face in dets:
    # 在图片中标注人脸，并显示
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    cv2.rectangle(img, (left, top), (right, bottom), (255 ,0, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)

    shape = predictor(img, face)  # 寻找人脸的68个标定点
    i = 1
    # 遍历所有点，打印出其坐标，并圈出来
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 1, ( 0,255, 0), 2)
        # 打印点的标号
        cv2.putText(img, str(i),pt_pos,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        i+=1
    cv2.imshow("image", img)
# 裁剪出人脸部分进行保存
imgsmall = img[ (top-50):(bottom+50),(left-50):(right+50),:]
cv2.imshow("image_small", imgsmall)
cv2.waitKey(0)
cv2.imwrite("Ronald68.jpg",imgsmall)
cv2.destroyAllWindows()
```

完整代码：[detectionDlib](./detectionDlib.py)





