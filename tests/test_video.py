import cv2

video_filepath = "C:/Users/Declan/Downloads/What Was I Made For_1080p.mp4"

cap = cv2.VideoCapture(video_filepath)

while(cap.isOpened()):
	ret, frame = cap.read()
	print(frame, ret)
	if ret:
		cv2.imshow("frame", frame)
		cv2.waitKey(1)
	else:
		break

cap.release()
cv2.destroyAllWindows()