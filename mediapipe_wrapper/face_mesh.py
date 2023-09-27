
import numpy as np

from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models
# https://github.com/googlesamples/mediapipe/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb

class FaceMesh:

	def __init__(self, model_filepath : str, num_of_faces=1):
		self.options = vision.FaceLandmarkerOptions( base_options=python.BaseOptions( model_asset_path=model_filepath ), num_faces=num_of_faces )
		self.detector = vision.FaceLandmarker.create_from_options(self.options)

	def detect( self, image : np.ndarray ) -> vision.FaceLandmarkerResult:
		return self.detector.detect( image )

	def visualize(
			self,
			image : np.ndarray,
			detection_result : vision.FaceLandmarkerResult,
			facemesh_tesselations=True,
			facemesh_contours=True,
			facemesh_irises=True
		) -> np.ndarray:
			annotated_image = image.copy()

			for face_landmarks in detection_result.face_landmarks:
				face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
				face_landmarks_proto.landmark.extend([
					landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
					for landmark in face_landmarks
				])

				if facemesh_tesselations:
					solutions.drawing_utils.draw_landmarks(
						image=annotated_image,
						landmark_list=face_landmarks_proto,
						connections=solutions.face_mesh.FACEMESH_TESSELATION,
						landmark_drawing_spec=None,
						connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style()
					)

				if facemesh_contours:
					solutions.drawing_utils.draw_landmarks(
						image=annotated_image,
						landmark_list=face_landmarks_proto,
						connections=solutions.face_mesh.FACEMESH_CONTOURS,
						landmark_drawing_spec=None,
						connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style()
					)

				if facemesh_irises:
					solutions.drawing_utils.draw_landmarks(
						image=annotated_image,
						landmark_list=face_landmarks_proto,
						connections=solutions.face_mesh.FACEMESH_IRISES,
						landmark_drawing_spec=None,
						connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
					)

			return annotated_image
