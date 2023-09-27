
import numpy as np

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
# https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb

class PoseTracking:

	def __init__( self, model_filepath : str ):
		self.options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_filepath),)
		self.detector = vision.PoseLandmarker.create_from_options(self.options)

	def detect( self, image : np.ndarray ) -> vision.PoseLandmarkerResult:
		return self.detector.detect( image )

	def visualize(
			self,
			image : np.ndarray,
			detection_result : vision.PoseLandmarkerResult
		) -> np.ndarray:

		annotated_image = image.copy()

		for pose_landmarks in detection_result.pose_landmarks:
			pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			pose_landmarks_proto.landmark.extend([
				landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
				for landmark in pose_landmarks
			])

			solutions.drawing_utils.draw_landmarks(
				annotated_image,
				pose_landmarks_proto,
				solutions.pose.POSE_CONNECTIONS,
				solutions.drawing_styles.get_default_pose_landmarks_style()
			)

		return annotated_image
