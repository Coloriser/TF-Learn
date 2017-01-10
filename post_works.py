import numpy as np
import skimage.color as color
import skimage.io as io
import pickle

def reconstruct(l_arr,a_arr):

	b_arr = np.zeros( shape = l_arr.shape)
	l_arr = np.zeros( shape = l_arr.shape)

	img = np.vstack(([l_arr.T], [a_arr.T], [b_arr.T])).T
	rgb_image = color.lab2rgb(img)

	io.imsave("recon_fake.jpg", rgb_image)


# For Testing ...

file_name = 'small_col.jpg'

img_rgb = io.imread(file_name)	
img_lab = color.rgb2lab(img_rgb)

l_arr = img_lab[:,:,0]

print("l_arr.shape : ",l_arr.shape)


f = open('predicted.chroma', 'r')
a_arr = pickle.load(f)
f.close()

print "a_arr \n \n \n "

print a_arr

real_a_arr = img_lab[:,:,1]

print "real_a_arr \n \n \n "
print real_a_arr

reconstruct(l_arr, a_arr)