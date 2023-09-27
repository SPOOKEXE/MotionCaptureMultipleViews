
import os
import requests

FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_STORE_PATH = os.path.join( FILE_DIRECTORY, "models" )

# Models available under:
# https://developers.google.com/mediapipe/solutions/guide

class FaceMeshModels:
	face_landmarker = (
		'face_landmarker.task',
		'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
	)

class FaceDetectionModels:
	face_detection_short_range = (
		'face_detector.tflite',
		'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'
	)

class ObjectDetectorModels:
	efficientdet = (
		"object_detector.tflite",
		"https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
	)

class PoseTrackingModels:
	pose_tracker_lite = (
		'pose_landmarker_lite.task',
		'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
	)
	pose_tracker_full = (
		'pose_landmarker_full.task',
		'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
	)
	pose_tracker_heavy= (
		'pose_landmarker_heavy.task',
		'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
	)

class HandMeshModels:
	hand_mesh = (
		'hand_landmarker.task',
		'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
	)

def get_model_filepath( model_tuple : tuple, directory=DEFAULT_MODEL_STORE_PATH ) -> str | None:
	# check if model is inside models dictionary
	assert type(model_tuple) == tuple, "Passed model data is not a tuple. Use the MODELS enumeration inside this file."

	filename, download_url = model_tuple

	# make sure directory has necessary ancestor folders
	os.makedirs( directory, exist_ok=True )

	# check if model exists in directory
	filepath = os.path.join( directory, filename )
	if os.path.exists( filepath ):
		return filepath

	try:
		with open( filepath, "wb" ) as file:
			response = requests.get( download_url )
			file.write( response.content )
		return filepath
	except Exception as exception:
		print(exception)
		return None
