import numpy as np
import skimage.color as color
import skimage.io as io
import pickle
def gray_scale(file_name):
	img_rgb = io.imread(file_name)
	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,0] # pull out L channel
	pickle.dump(img_l, "gray.pickle")

gray_scale('small_col.jpg')