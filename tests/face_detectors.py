
import cv2
import keyboard

from os import path as os_path
from sys import path as sys_path

FILE_DIRECTORY = os_path.dirname(os_path.realpath(__file__))

sys_path.append( os_path.join( FILE_DIRECTORY, ".." ) )

from mediapipe_wrapper import (
	FaceDetection,
	FaceDetectionModels,
	FaceMesh,
	FaceMeshModels,
	PoseTracking,
	PoseTrackingModels,
	HandMesh,
	HandMeshModels,
	image,
	get_model_filepath,
)

sys_path.pop()

if __name__ == '__main__':

	CAMERA_RESOLUTION = (720, 480)

	print("Loading Models & Detectors")

	face_detection = FaceDetection( get_model_filepath( FaceDetectionModels.face_detection_short_range ) )
	face_mesh = FaceMesh( get_model_filepath( FaceMeshModels.face_landmarker ), num_of_faces=1 )
	hand_mesh = HandMesh( get_model_filepath( HandMeshModels.hand_mesh ), num_of_hands=2 )
	pose_tracker = PoseTracking( get_model_filepath( PoseTrackingModels.pose_tracker_full ) )

	print("Finished Loading Models & Detectors")

	vid = cv2.VideoCapture( 0, cv2.CAP_DSHOW )
	vid.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
	vid.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

	# Switch which detectors are enabled and are visualized
	# 0 = None
	# 1 = All
	# 2 = Face Detection
	# 3 = Face mesh
	# 4 = Hand Mesh
	# 5 = Pose Detection
	# 6 = Face Detectors
	# 7 = Body Detectors
	mode = modeName = None
	modeKeyMatrix = [ "1", "2", "3", "4", "5", "6", "7", "8" ]
	def switch_mode( target_mode : int ) -> None:
		global mode, modeName
		if mode == target_mode:
			return
		mode = target_mode
		if target_mode == 0:
			modeName = "None"
		elif target_mode == 1:
			modeName = "All Detectors"
		elif target_mode == 2:
			modeName = "Face Detection"
		elif target_mode == 3:
			modeName = "Face Mesh"
		elif target_mode == 4:
			modeName = "Hand Mesh"
		elif target_mode == 5:
			modeName = "Pose Detector"
		elif target_mode == 6:
			modeName = "Face Detectors"
		elif target_mode == 7:
			modeName = "Body Detectors"
		print( modeName )
	switch_mode( 0 )

	while True:
		# break loop
		if keyboard.is_pressed('q'):
			break

		# raw feed
		ret, frame = vid.read()
		# cv2.imshow('Raw Feed', frame)

		# annotated feed
		mp_image_raw = image.cv2_to_mediapipe_image( frame )
		annotated_image = frame.copy()

		# face box
		if mode == 1 or mode == 2 or mode == 6:
			result = face_detection.detect( mp_image_raw )
			annotated_image = face_detection.visualize( annotated_image, result )

		# face mesh
		if mode == 1 or mode == 3 or mode == 6:
			result = face_mesh.detect( mp_image_raw )
			annotated_image = face_mesh.visualize( annotated_image, result )

		# hand mesh
		if mode == 1 or mode == 4 or mode == 7:
			result = hand_mesh.detect( mp_image_raw )
			annotated_image = hand_mesh.visualize( annotated_image, result )

		# pose detection
		if mode == 1 or mode == 5 or mode == 7:
			result = pose_tracker.detect( mp_image_raw )
			annotated_image = pose_tracker.visualize( annotated_image, result )

		# place text ontop & display feed
		cv2.putText( annotated_image, modeName, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1, 2, bottomLeftOrigin=False )
		cv2.imshow('Detector Feed', annotated_image)

		# mode toggle
		for index, keyItem in enumerate(modeKeyMatrix):
			if keyboard.is_pressed(keyItem):
				switch_mode( index )
				break

		# delay for key presses
		cv2.waitKey(1)

	vid.release()
	cv2.destroyAllWindows()
