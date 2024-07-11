from scipy.spatial import distance  # scipy.spatial 모듈에서 유클리드 거리 계산 함수를 가져옴
from imutils import face_utils  # imutils 모듈에서 얼굴 랜드마크 유틸리티를 가져옴
import imutils  # 이미지 처리에 유용한 imutils 모듈을 가져옴
import dlib  # 얼굴 인식 및 랜드마크 검출을 위한 dlib 라이브러리 가져옴
import cv2  # 컴퓨터 비전 작업을 위한 OpenCV 라이브러리 가져옴

def eye_aspect_ratio(eye):  # 눈 비율을 계산하는 함수 정의
	A = distance.euclidean(eye[1], eye[5])  # 눈의 두 점 사이의 유클리드 거리 계산
	B = distance.euclidean(eye[2], eye[4])  # 눈의 다른 두 점 사이의 유클리드 거리 계산
	C = distance.euclidean(eye[0], eye[3])  # 눈의 가로 길이 계산
	ear = (A + B) / (2.0 * C)  # 눈 비율 계산
	return ear  # 계산된 눈 비율 반환
	
thresh = 0.25  # 눈 비율 임계값 설정
frame_check = 20  # 프레임 체크 기준 설정
detect = dlib.get_frontal_face_detector()  # dlib의 정면 얼굴 탐지기 초기화
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # 랜드마크 예측 모델 불러오기

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]  # 왼쪽 눈 랜드마크 인덱스
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]  # 오른쪽 눈 랜드마크 인덱스
cap = cv2.VideoCapture(0)  # 웹캠 초기화
flag = 0  # 깜박임을 체크하기 위한 플래그 초기화

while True:  # 무한 루프 시작
	ret, frame = cap.read()  # 웹캠에서 프레임 읽어오기
	frame = imutils.resize(frame, width=450)  # 프레임 크기 조정
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 그레이스케일로 변환
	subjects = detect(gray, 0)  # 얼굴 탐지

	for subject in subjects:  # 탐지된 얼굴마다 반복
		shape = predict(gray, subject)  # 얼굴 랜드마크 예측
		shape = face_utils.shape_to_np(shape)  # 랜드마크를 NumPy 배열로 변환
		leftEye = shape[lStart:lEnd]  # 왼쪽 눈 랜드마크 추출
		rightEye = shape[rStart:rEnd]  # 오른쪽 눈 랜드마크 추출
		leftEAR = eye_aspect_ratio(leftEye)  # 왼쪽 눈 비율 계산
		rightEAR = eye_aspect_ratio(rightEye)  # 오른쪽 눈 비율 계산
		ear = (leftEAR + rightEAR) / 2.0  # 양쪽 눈 비율의 평균 계산
		leftEyeHull = cv2.convexHull(leftEye)  # 왼쪽 눈의 볼록 껍질 계산
		rightEyeHull = cv2.convexHull(rightEye)  # 오른쪽 눈의 볼록 껍질 계산
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # 왼쪽 눈 윤곽선을 프레임에 그림
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # 오른쪽 눈 윤곽선을 프레임에 그림
		if ear < thresh:  # 눈 비율이 임계값보다 작으면
			flag += 1  # 플래그 증가
			print(flag)  # 플래그 값 출력
			if flag >= frame_check:  # 플래그가 프레임 체크 기준 이상이면
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 경고 메시지 표시 (위쪽)
				cv2.putText(frame, "****************ALERT!****************", (10, 325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # 경고 메시지 표시 (아래쪽)
		else:  # 눈 비율이 임계값보다 크면
			flag = 0  # 플래그 초기화

	cv2.imshow("Frame", frame)  # 프레임을 화면에 표시
	key = cv2.waitKey(1) & 0xFF  # 키 입력 대기
	if key == ord("q"):  # 'q' 키가 입력되면
		break  # 루프 종료

cv2.destroyAllWindows()  # 모든 창 닫기
cap.release()  # 웹캠 해제
