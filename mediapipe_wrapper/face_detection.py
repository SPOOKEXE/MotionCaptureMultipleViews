import numpy as np
import cv2
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Union, Tuple

def _normalized_to_pixel_coordinates(
		normalized_x: float,
		normalized_y: float,
		image_width: int,
		image_height: int
	) -> Union[None, Tuple[int, int]]:

	def is_valid_normalized_value(value: float) -> bool:
		return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

	if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
		return None
	x_px = min(math.floor(normalized_x * image_width), image_width - 1)
	y_px = min(math.floor(normalized_y * image_height), image_height - 1)
	return x_px, y_px

# https://developers.google.com/mediapipe/solutions/vision/face_detector
# https://github.com/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb

class FaceDetection:

	def __init__( self, model_filepath : str ):
		print(model_filepath)
		self.options = vision.FaceDetectorOptions(base_options=python.BaseOptions(model_asset_path=model_filepath))
		self.detector = vision.FaceDetector.create_from_options(self.options)

	def detect( self, image : np.ndarray ) -> np.ndarray:
		return self.detector.detect(image)

	def visualize(
			self,
			image : np.ndarray,
			detection_result : vision.FaceDetectorResult,
			textColor = (255,255,255),
			showBox=True,
			showPoints=True,
			showScore=True
		) -> np.ndarray:
		annotated_image = image.copy()
		height, width, _ = image.shape

		MARGIN = 10  # pixels
		ROW_SIZE = 10  # pixels
		FONT_SIZE = 1
		FONT_THICKNESS = 1

		for detection in detection_result.detections:
			bbox = detection.bounding_box
			if showBox:
				start_point = bbox.origin_x, bbox.origin_y
				end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
				cv2.rectangle(annotated_image, start_point, end_point, textColor, 3)

			if showPoints:
				for keypoint in detection.keypoints:
					keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
					color, thickness, radius = (0, 255, 0), 2, 2
					cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

			if showScore:
				category = detection.categories[0]
				category_name = category.category_name
				category_name = '' if category_name is None else category_name
				probability = round(category.score, 2)
				result_text = category_name + ' (' + str(probability) + ')'
				text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
				cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, textColor, FONT_THICKNESS)

		return annotated_image