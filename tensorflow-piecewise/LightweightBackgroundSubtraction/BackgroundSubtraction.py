# BackgroundSubtractorMOG2 배경 제거 (track_bgsub_mog2.py)

import numpy as np, cv2, time

# video load
capture = cv2.VideoCapture('../../Data/AIC22_Track1_MTMC_Tracking/train/S03/c013/vdo.avi')
# single-channel region-of-interest mask image load
roi_mask = cv2.imread('../../Data/AIC22_Track1_MTMC_Tracking/train/S03/c013/roi.jpg', cv2.IMREAD_UNCHANGED)
roi_mask = cv2.resize(roi_mask, (854, 480), interpolation=cv2.INTER_CUBIC)

fps = capture.get(cv2.CAP_PROP_FPS) # 프레임 수 구하기
delay = int(600/fps)

# 배경 제거 객체 생성 --- ①
kernel = None
backgroundObject = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=128, detectShadows=False)
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    # resize : 이미지 크기 변환
    # 1) 변환할 이미지
    # 2) 변환할 이미지 크기(가로, 세로)
    # - interpolation : 보간법 지정
    #   - 보간법 : 알려진 데이터 지점 내에서 새로운 데이터 지점을 구성하는 방식
    #   - cv2.INTER_NEAREST : 최근방 이웃 보간법
    #   - cv2.INTER_LINEAR(default) : 양선형 보간법(2x2 이웃 픽셀 참조)
    #   - cv2.INTER_CUBIC : 3차 회선 보간법(4x4 이웃 픽셀 참조)
    #   - cv2.INTER_LANCZOS4 : Lanczos 보간법(8x8 이웃 픽셀 참조)
    #   - cv2.INTER_AREA : 픽셀 영역 관계를 이용한 resampling 방법으로 이미지 축소시 효과적
    frame = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_CUBIC)

    # 배경 제거 마스크 계산
    took = time.time()
    foreground_mask = cv2.bitwise_and(frame, frame, mask=roi_mask)
    foreground_mask = backgroundObject.apply(foreground_mask)
    _, foreground_mask = cv2.threshold(foreground_mask, 250, 255, cv2.THRESH_BINARY)
    foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
    foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=10)
    print("mask {:.5f} ms".format((time.time() - took) * 1000))
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameCopy = frame.copy()
    # loop over each contour found in the frame.
    for cnt in contours:
        # We need to be sure about the area of the contours i.e. it should be higher than 256 to reduce the noise.
        if cv2.contourArea(cnt) > 256:
            # Accessing the x, y and height, width of the objects
            x, y, width, height = cv2.boundingRect(cnt)    
            # Here we will be drawing the bounding box on the objects
            cv2.rectangle(frameCopy, (x , y), (x + width, y + height),(0, 0, 255), 2)
            # Then with the help of putText method we will write the 'detected' on every object with a bounding box
            cv2.putText(frameCopy, 'Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

            # todo: YOLO

    print("detect {:.5f} ms".format((time.time() - took) * 1000))
    foregroundPart = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    frame = np.hstack((frame, foregroundPart, frameCopy))

    cv2.imshow('frame',frameCopy)
    cv2.imshow('foregroundPart',foregroundPart)
    if cv2.waitKey(delay) & 0xff == 27:
        break
capture.release()
cv2.destroyAllWindows()