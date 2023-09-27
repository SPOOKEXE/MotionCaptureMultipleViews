
import cv2
import numpy as np
import mediapipe as mp

from PIL import Image

# PIL Image
def PIL_to_cv2_image( pil_img : Image.Image ) -> np.ndarray:
	return cv2.cvtColor( np.asarray(pil_img) , cv2.COLOR_RGB2BGR )

def PIL_to_mediapipe_image( pil_img : Image.Image ) -> mp.Image:
	return mp.Image( image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img) )

# cv2
def cv2_to_PIL_image( cv2_img : np.ndarray ) -> Image.Image:
	return Image.fromarray( cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB) )

def cv2_to_mediapipe_image( cv2_img : np.ndarray ) -> mp.Image:
	return mp.Image( image_format=mp.ImageFormat.SRGB, data=cv2_img )

# mediapipe

# def mediapipe_to_PIL_image( mp_img : mp.Image ) -> Image.Image:
# 	pass

# def mediapipe_to_cv2_image( mp_img : mp.Image ) -> np.ndarray:
# 	pass
