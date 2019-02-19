import cv2
import dlib
import numpy as np
from PIL import Image



def get_eye(img_src,record, record1,record2):

    img = cv2.imread(f"{img_src}.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        #右眼睛区域左上角点(x,y)，右下角点(x0,y0)
        x, y, x0, y0 = landmarks.part(42).x - 5, min(landmarks.part(43).y, landmarks.part(44).y) - 5, landmarks.part(45).x + 5, max(landmarks.part(46).y, landmarks.part(47).y) + 5
        #左眼睛区域左上角点(x1,y1)，右下角点(x2,y2)
        x1,y1,x2,y2 = landmarks.part(36).x-5,min(landmarks.part(37).y,landmarks.part(38).y)-5,landmarks.part(39).x+5,max(landmarks.part(40).y,landmarks.part(41).y)+5

        # 嘴巴区域左上角点(x3,y3)，右下角点(x4,y4)
        x3, y3, x4, y4 = landmarks.part(48).x-5, min(landmarks.part(50).y,landmarks.part(52).y)-5, landmarks.part(54).x+5,landmarks.part(57).y +5

        cut = img[y:y0,x:x0]
        cut1 = img[y1:y2,x1:x2]
        cut2 = img[y3:y4,x3:x4]

        record[img_src] = (x, y, x0, y0)
        record1[img_src] = (x1,y1,x2,y2)
        record2[img_src] = (x3,y3,x4,y4)

        return cut,cut1,cut2

def changeface(p1,p2):
    record={}
    record1={}
    record2={}

    eye_right, eye_left, mouth1 = get_eye(p1,record,record1,record2)
    eye_right2, eye_left2, mouth2 = get_eye(p2,record,record1,record2)

    eye_right_resized = cv2.resize(eye_right,(record[p2][2]-record[p2][0],record[p2][3]-record[p2][1]),interpolation=cv2.INTER_AREA)
    eye_left_resized = cv2.resize(eye_left, (record1[p2][2] - record1[p2][0], record1[p2][3] - record1[p2][1]),interpolation=cv2.INTER_AREA)

    eye_right2_resized = cv2.resize(eye_right2,(record[p1][2]-record[p1][0],record[p1][3]-record[p1][1]),interpolation=cv2.INTER_AREA)
    eye_left2_resized = cv2.resize(eye_left2, (record1[p1][2] - record1[p1][0], record1[p1][3] - record1[p1][1]),interpolation=cv2.INTER_AREA)

    mouth1_resized = cv2.resize(mouth1,(record2[p2][2]-record2[p2][0],record2[p2][3]-record2[p2][1]),interpolation=cv2.INTER_AREA)
    mouth2_resized = cv2.resize(mouth2, (record2[p1][2] - record2[p1][0], record2[p1][3] - record2[p1][1]),interpolation=cv2.INTER_AREA)

    im = cv2.imread(f"{p1}.jpg")
    obj = eye_right2_resized
    # Create an all white mask
    mask = 255 * np.ones(obj.shape, obj.dtype)
    center = (int((record[p1][0] + record[p1][2]) / 2), int((record[p1][1] + record[p1][3]) / 2))
    # Seamlessly clone src into dst and put the results in output
    normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)

    im1 = normal_clone
    obj1 = eye_left2_resized
    # Create an all white mask
    mask1 = 255 * np.ones(obj1.shape, obj1.dtype)
    center1 = (int((record1[p1][0] + record1[p1][2]) / 2), int((record1[p1][1] + record1[p1][3]) / 2))
    # Seamlessly clone src into dst and put the results in output
    normal_clone1 = cv2.seamlessClone(obj1, im1, mask1, center1, cv2.NORMAL_CLONE)

    im2 = normal_clone1
    obj2 = mouth2_resized
    mask2 = 255 * np.ones(obj2.shape, obj2.dtype)
    center2 = (int((record2[p1][0] + record2[p1][2]) / 2), int((record2[p1][1] + record2[p1][3]) / 2))
    normal_clone2 = cv2.seamlessClone(obj2, im2, mask2, center2, cv2.NORMAL_CLONE)
    cv2.imwrite(f"{p1}_result.jpg", normal_clone2)


# src1 = input("请输入第一张图片名称，如baby.jpg请输入baby,无需后缀：")
# src2 = input("请输入第二张图片名称，如baby.jpg请输入baby,无需后缀：")

src1,src2 = "huge","heben"

#add src2's eyes and mouth to src1's face
changeface(src1,src2)
#add src2's eyes and mouth to src1's face
changeface(src2,src1)
