import cv2
import os
import imutils
import numpy as np
import math

global x, y, angle, scale

def template_matching(img_template, img_reference):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #mask test용
    mask = cv2.imread("mask.png")
    for angle in range(0, 360, 10):
        for scale in range(1, 4, 1):
            #scale 를 0.5, 1, 1.5 배로 테스트
            scale = scale/2
            #scale 변환
            template = cv2.resize(img_template, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            #회전
            rotated = imutils.rotate_bound(template, angle)

            #마스크 생성
            #mask_t = cv2.resize(mask, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            #mask_r = imutils.rotate_bound(mask_t, angle)

            #실행 해보니 흰색 부분만 찾음
            #문제점 - 흰색만 찾는 것으로 보아 mask를 사용하면 threshold로 인해 하얀 이미지가 되는 것 같음
            #해결 방안 - threshold의 비율을 조정하여 흰색의 이미지만 따로 처리하는 식으로 하면 될 것 같은데 방법을 못찾았음
            #res = cv2.matchTemplate(img_reference, rotated, cv2.TM_CCOEFF_NORMED, mask = mask_r)
            
            #template matching
            res = cv2.matchTemplate(img_reference, rotated, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.7)

            #리턴
            if(len(loc[0]) > 0):
                for pt in zip(*loc[::-1]):
                    return pt[0], pt[1], angle, scale
    return 0, 0, 0, 0