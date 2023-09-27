
import numpy as np
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
# https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

class HandMesh:

	def __init__(self, model_filepath : str, num_of_hands=2 ):
		self.options = vision.HandLandmarkerOptions( base_options=python.BaseOptions(model_asset_path=model_filepath), num_hands=num_of_hands )
		self.detector = vision.HandLandmarker.create_from_options(self.options)

	def detect( self, image : np.ndarray ) -> vision.HandLandmarkerResult:
		return self.detector.detect( image )

	def visualize(
			self,
			image : np.ndarray,
			detection_result : vision.HandLandmarkerResult,
			hand_connections=True,
			hand_info=True,
		) -> np.ndarray:

		MARGIN = 10 # pixels
		FONT_SIZE = 1
		FONT_THICKNESS = 1
		HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

		annotated_image = image.copy()

		# Loop through the detected hands to visualize.
		for hand_landmarks, handedness in zip(detection_result.hand_landmarks, detection_result.handedness):
			hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			hand_landmarks_proto.landmark.extend([
				landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
				for landmark in hand_landmarks
			])

			if hand_connections:
				solutions.drawing_utils.draw_landmarks(
					annotated_image,
					hand_landmarks_proto,
					solutions.hands.HAND_CONNECTIONS,
					solutions.drawing_styles.get_default_hand_landmarks_style(),
					solutions.drawing_styles.get_default_hand_connections_style()
				)

			if hand_info:
				height, width, _ = annotated_image.shape
				x_coordinates = [landmark.x for landmark in hand_landmarks]
				y_coordinates = [landmark.y for landmark in hand_landmarks]
				text_x = int(min(x_coordinates) * width)
				text_y = int(min(y_coordinates) * height) - MARGIN

				cv2.putText(
					annotated_image,
					f"{handedness[0].category_name}",
					(text_x, text_y),
					cv2.FONT_HERSHEY_DUPLEX,
					FONT_SIZE,
					HANDEDNESS_TEXT_COLOR,
					FONT_THICKNESS,
					cv2.LINE_AA
				)

		return annotated_image