import cv2
import numpy as np
import glob
import sys
from PIL import ImageFont, ImageDraw,Image
#이미지에 한글 입력을 위한 import
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0) #캠 영상 불러오기
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)
# 영상 사이즈 지정하기
(x,y,w,h) = (600,300,350,350)
#손금을 캡처할 영역의 좌표를 설정해 줍니다
if cap.isOpened():
    ret,frame = cap.read() #카메라 프레임 읽기
    while ret:
        ret, frame = cap.read()
        cv2.putText(frame,"Put your palm of the left hand on the area. ",(450,220),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255))
        cv2.putText(frame, "Tap the keyboard 'c' to capture it. ", (500, 260), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        #화면에 메시지 표시하기
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
        #손바닥을 캡처할 영역을 화면에 표시합니다
        cv2.imshow('camera',frame)
        # roi로 지정할 사각형 영역을 카메라 화면에표시
        if ret:
            key = cv2.waitKey(1)
            if key == 27:  # esc 프로그램 종료
                sys.exit()
            elif key ==ord('c'):  #c 키를 눌러야 화면 캡처
                cv2.imwrite('img/cam_cap.jpg',frame) ##현재화면을 캡처합니다
                cv2.destroyWindow('camera')
                break

##########ROI 지정후 역투영 코드 시작################
win_name = 'project'
img = cv2.imread('./img/cam_cap.jpg') #저장된 이미지를 불러옵니다
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #HSV 컬러 스페이스

# 역투영된 결과를 마스킹해서 결과를 출력하는 함수
def masking(bp, win_name):
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(bp, -1, disc, bp)
    _, mask = cv2.threshold(bp, 1, 255, cv2.THRESH_BINARY)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite('img/resulthist.jpg',result)
    cv2.imshow('result',result) #########
    #resulthist로 역투영 결과 저장

#  역투영 함수
def backProject_manual(hist_roi):
    # 전체 영상에 대한 H,S 히스토그램 계산
    hist_img = cv2.calcHist([hsv_img], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 선택영역과 전체 영상에 대한 히스토그램 그램 비율계산
    hist_rate = hist_roi / (hist_img + 1)
    # 비율에 맞는 픽셀 값 매핑
    h, s, v = cv2.split(hsv_img)
    bp = hist_rate[h.ravel(), s.ravel()]
    bp = np.minimum(bp, 1)
    bp = bp.reshape(hsv_img.shape[:2])
    cv2.normalize(bp, bp, 0, 255, cv2.NORM_MINMAX)
    bp = bp.astype(np.uint8)
    # 투영 결과로 마스킹해서 결과 출력
    masking(bp, 'result_manual')

roi = frame[y:y+h, x:x+w]
roi2 = frame[y+50:y+h-50, x+50:x+w-50]
#roi영역보다 작은 영역을 역투영해서 역투영 결과를 좀더 정확하게 해줌
hsv_roi = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
hist_roi = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
backProject_manual(hist_roi)
##########ROI 지정후 역투영 코드 끝################

img = cv2.imread('img/resulthist.jpg')
#저장된 역투영 결과를 불러옵니다
img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img2 = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
cv2.imshow("wait....",img2)
cv2.waitKey(1000)
#선명하게 해주기 위해서 역투영 결과 사진을 이퀄라이즈 해줍니다

gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#경계검출을 위해서 그레이스케일 이미지로 변경
cv2.imshow('wait....',gray)
cv2.waitKey(1000)

edges = cv2.Canny(gray, 90,110, apertureSize=3)
#캐니 엣지 알고리즘으로 엣지 검출을 합니다
cv2.imshow('wait....',edges)
cv2.waitKey(1000)

edges = cv2.bitwise_not(edges)
#경계를 검은색으로 지정하기 위해서 bitwise_not 연산을 수행합니다
cv2.imshow('wait....',edges)
cv2.waitKey(1000)
cv2.imwrite("palmlines.jpg", edges)
##변환 이미지를 저장합니다

### 이미지 기존 이미지에 합성 - 마지막 작업
palmlines = cv2.imread("palmlines.jpg")
img = cv2.addWeighted(palmlines, 0.3, img, 0.7, 0)
img3 = cv2.addWeighted(img ,0.3, img2 ,0.7,0)
cv2.imshow("wait....", img3)
cv2.waitKey(1000)

img4 = cv2.addWeighted(img2 ,0.3, img3 ,0.7,0)
result_img = cv2.addWeighted(img3 ,0.3, img4 ,0.7,0)
#반복 연산으로 손금을 더선명하게 시도
cv2.imshow("wait....", result_img)
cv2.waitKey(1000)
cv2.destroyWindow("wait....")

sharpening_2 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
result_img = cv2.filter2D(result_img, -1, sharpening_2)
#샤프닝 적용 - 경계를 좀더 날카롭게
cv2.imshow("result!", result_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
#최종 결과를 저장
roi_copy = result_img[y:y+h, x:x+w]
result_roi = roi_copy.copy()
#이미지 비교를 위해서 최종 결과의 손바닥 영역을 카피
cv2.imwrite('img/resulthist.jpg',result_img)
result_img_wait_mess = result_img.copy()
#검사 진행중을 임시로  text로 표현하기 위해서 카피해줌
cv2.putText(result_img_wait_mess, "OK!Please wait. ", (200, 500), cv2.FONT_HERSHEY_TRIPLEX, 4, (255, 255, 255))
cv2.imshow("result!", result_img_wait_mess)
cv2.waitKey(2000)
#잠시 기다리라는 메시지 출력 후 종료

###### 매칭 영역##################

########비교 이미지 생성 코드############
### roi영역에 손금 비교 이미지를 합성하기 위해서 기존 이미지 블러링

background = cv2.blur(roi, (10, 10))
# 비교 이미지를 만들기 위해서 백그라운드 영역을 블러링 처리를 해줍니다
# 손금영역을 최대한 지운상태에서 기존 손금 이미지들과 합성하면서 가장 적합한 손금 찾기
max_accuracy = 0.0 #가장 매칭 정확도가 높은 값을 얻기 위한 변수
max_num = 0  #가장 매칭 정확도가 높은 값의 인덱스를 얻기 위한 변수

#매칭 탐색 함수
def find_math(line_img_name_a):
    global max_num
    global max_accuracy
    cv2.imshow("result_roi...find...match",result_roi)
    for i in range(1,4):
        line_img_name = line_img_name_a+str(i)+'.jpg'
        line_img = cv2.imread(line_img_name,cv2.IMREAD_UNCHANGED)
        #저장 되어 있는 손금 이미지를 불러옵니다
        gray_line_img = cv2.cvtColor(line_img,cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray_line_img, 10, 255, cv2.THRESH_BINARY_INV)
        #합성할 mask를 생성합니다 -> 바이너리 이미지
        added = cv2.add(line_img,background,mask=mask_inv)
        result_line = cv2.add(added,line_img) #기존의 손금 이미지를 사용자의 손바닥에 합성

        #이미지를 평균 해시로 변환
        def img2hash(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (350,350)) #이미지를 350*350의 평균 해시로 변환
            avg = gray.mean()
            bi  = 1*(gray>avg)
            return bi
        #해밍 거리 측정 함수
        def hamming_distance(a,b):
            a = a.reshape(1,-1)
            b = b.reshape(1,-1)
            #같은 자리의 값이 서로 다른 것들의 합
            distance = (a!=b).sum()
            return distance

        #기존 이미지의 해시 구하기
        query_hash = img2hash(result_roi)
        #합성 이미지의 해시 구하기
        a_hash = img2hash(result_line)
        #해밍 거리 산출
        dst = hamming_distance(query_hash,a_hash)
        match_val = dst/(350*350) #최대값 비교를 위해서 유사율 저장
        print(line_img_name+" : "+str(match_val)) #해당 값 출력
        cv2.imshow('mathing......',result_line)
        cv2.waitKey(700)
        if(match_val > max_accuracy):
            max_accuracy = match_val
            max_num = i
         #매칭률이 제일 큰 인덱스와 값을 저장하는 코드
    cv2.destroyAllWindows()
    return max_num

######################감정선 분석##############
max_accuracy = 0.0
max_num = 0
#매칭 측정값 초기화
line_img_name_a = 'img/emotion_line/line'
emotion_match_num = find_math(line_img_name_a) #감정선 매칭 탐색
#return 값으로 가장 매칭률이 높은 인덱스가 나온다
line_img_name = 'img/emotion_line/line'+str(emotion_match_num)+'.jpg'
#해당 인덱스로 파일 불러오기
line_img = cv2.imread(line_img_name)
if(emotion_match_num==1):
    emotion_text = "소극적이며,내성적,자심감 결여 상태"
    emotion_text2 = "애정 표현에 서툽니다"
elif(emotion_match_num==2):
    emotion_text = "냉담한 성격,개인주의적 성향이 강함"
    emotion_text2 = "이성적인 모습에 신뢰성은 높음"
else:
    emotion_text = "정신적, 육체적 사랑 모두 중요시"
    emotion_text2 = "사랑 앞에서 나이를 따지지 않음"
##이미지에 분석 내용 보여주기
font = ImageFont.truetype("fonts/gulim.ttc", 15)
result_img = Image.fromarray(result_img)
draw = ImageDraw.Draw(result_img)
text = "감정선 분석 완료"
draw.text((1000, 300), text, font=font, fill=(0, 255, 0))
draw.text((1000, 350), emotion_text, font=font, fill=(255, 255, 255))
draw.text((1000, 400), emotion_text2, font=font, fill=(255, 255, 255))
result_img =np.array(result_img)
cv2.waitKey(700)
#######################################
######################두뇌선 분석##############
max_accuracy = 0.0
max_num = 0
#매칭 측정값 초기화
line_img_name_a = 'img/brain_line/line'
brain_match_num = find_math(line_img_name_a) #두뇌선 매칭 탐색
#return 값으로 가장 매칭률이 높은 인덱스가 나온다
line_img_name = 'img/brain_line/line'+str(brain_match_num)+'.jpg'
#해당 인덱스로 파일 불러오기
line_img = cv2.imread(line_img_name)

if(brain_match_num==1):
    brain_text = "착실한 성격 "
    brain_text2 = "과격한 행동을 하지 않음"
elif(brain_match_num==2):
    brain_text = "미술,음악에 적성에 맞음"
    brain_text2 = "예술적 감각이 뛰어남"
else:
    brain_text = "자기 관리가 철저함"
    brain_text2 = "말년에 고독할 팔자 "
##이미지에 분석 내용 보여주기
font = ImageFont.truetype("fonts/gulim.ttc", 15)
result_img = Image.fromarray(result_img)
draw = ImageDraw.Draw(result_img)
text = "두뇌선 분석 완료"
draw.text((200, 300), text, font=font, fill=(0, 255, 0))
draw.text((200, 350), brain_text, font=font, fill=(255, 255, 255))
draw.text((200, 400), brain_text2, font=font, fill=(255, 255, 255))
result_img =np.array(result_img)
cv2.waitKey(700)

######################생명선 분석##############
max_accuracy = 0.0
max_num = 0
#매칭 측정값 초기화
line_img_name_a = 'img/life_line/line'
life_match_num = find_math(line_img_name_a) #두뇌선 매칭 탐색
#return 값으로 가장 매칭률이 높은 인덱스가 나온다
line_img_name = 'img/life_line/line'+str(life_match_num)+'.jpg'
#해당 인덱스로 파일 불러오기
line_img = cv2.imread(line_img_name)
if(life_match_num==1):
    life_text = "길고 선명한 생명선"
    life_text2 = "무병 장수할 운명"
elif(life_match_num==2):
    life_text = "타고난 천수를 다 누리지 못할 팔자"
    life_text2 = "사고를 조심해야 함 "
else:
    life_text = "각별히 몸조심"
    life_text2 = "40대 이후 건강 중요"
##이미지에 분석 내용 보여주기
font = ImageFont.truetype("fonts/gulim.ttc", 15)
result_img = Image.fromarray(result_img)
draw = ImageDraw.Draw(result_img)
text = "생명선 분석 완료"
draw.text((200, 450), text, font=font, fill=(0, 255, 0))
draw.text((200, 500), life_text, font=font, fill=(255, 255, 255))
draw.text((200, 550), life_text2, font=font, fill=(255, 255, 255))
result_img =np.array(result_img)
cv2.waitKey(700)

##################최종 화면####################

cv2.putText(result_img, "Check out the results!", (150, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255))
cv2.imshow("final_result",result_img)
cv2.imwrite("img/final_result.jpg",result_img)
cv2.waitKey()
cv2.destroyAllWindows()