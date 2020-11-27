import cv2
import os
import hw02_2018315021_template_matching
import imutils
import numpy as np
import math

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    #이미지 읽기
    img_template = cv2.imread('../img/fish.png')
    img_reference = cv2.imread('../test_background.png')
    img = cv2.imread('../test_background.png')
    
    #template_matching 함수 호출
    x, y, angle, scale = hw02_2018315021_template_matching.template_matching(img_template, img_reference)

    #사각형으로 표현할 크기 설정
    rotated = imutils.rotate_bound(img_template, angle)
    h = rotated.shape[0] * scale
    w = rotated.shape[1] * scale

    #sin과 cos 사용하여 포인트 찾고, 사각형 표시
    if(angle != 0):
        if(angle < 90 or (angle > 180 and angle < 270)):
            pts = np.array([[x + abs(math.sin(math.pi/(18/(angle/10))) * (37 * scale)), y], [x + w, y + abs(math.sin(math.pi/(18/(angle/10))) * (85 * scale))], 
            [x + abs(math.cos(math.pi/(18/(angle/10))) * (85 * ((i+1)/2))), y + h], [x,y + abs(math.cos(math.pi/(18/(angle/10))) * (37 * ((i+1)/2)))]], np.int32)
            cv2.polylines (img, [pts], True , (0,255,0), 2)
        elif(angle > 270 or (angle > 90 and angle < 180)):
            angle_test = angle - 180
            pts = np.array([[x + abs(math.cos(math.pi/(18/(angle_test/10))) * (85 * scale)), y], [x + w, y + abs(math.cos(math.pi/(18/(angle_test/10))) * (37 * scale))], 
            [x + abs(math.sin(math.pi/(18/(angle_test/10))) * (37 * scale)), y + h], [x,y + abs(math.sin(math.pi/(18/(angle_test/10))) * (85 * scale))]], np.int32)
            cv2.polylines (img, [pts], True , (0,255,0), 2)

    cv2.imshow("A", img)
    cv2.waitKey(0)
main()