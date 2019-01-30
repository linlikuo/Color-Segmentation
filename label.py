from roipoly import RoiPoly
from matplotlib import pyplot as plt
from matplotlib import image
import cv2, os, pickle
import numpy as np
import argparse

folder = './trainset'

parser = argparse.ArgumentParser()
parser.add_argument('--item', type=str, default = 'barrel_blue', help = '{barrel_blue, not_barrel_blue, brown, green}')

args = parser.parse_args()
print(args)

def label(item_name = 'barrel_blue', color_space = 'hsv'):
	'''4 class = {barrel_blue, not_barrel_blue, brown, green}
	color_space = ('rgb', 'hsv', 'ycrcb')'''
	info_rgb = np.empty((1,3))
	info_hsv = np.empty((1,3))
	info_ycrcb = np.empty((1,3))
	
	for im_file in os.listdir(folder):
		plt.figure()
		img_bgr = cv2.imread(os.path.join(folder, im_file))
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		plt.imshow(img_rgb)
		Roi1 = RoiPoly(color='r')
		
		# show the image with the first ROI
		#plt.imshow(img_rgb)
		#Roi1.display_roi()
	
		# show Roi masks
		#plt.imshow(Roi1.get_mask(img_rgb))
		#plt.show()	
		
		# feature generate
		mask = Roi1.get_mask(img_rgb)
		point_interest = np.where(mask == True)

		# HSV color space		
		img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)		
		
		fea_hsv_0 = img_hsv[:,:,0]
		fea_hsv_1 = img_hsv[:,:,1]
		fea_hsv_2 = img_hsv[:,:,2]

		F_hsv_0 = fea_hsv_0[point_interest]
		F_hsv_1 = fea_hsv_1[point_interest]
		F_hsv_2 = fea_hsv_2[point_interest]

		temp_hsv = np.column_stack((F_hsv_0, F_hsv_1, F_hsv_2))
		info_hsv = np.concatenate((info_hsv, temp_hsv), axis = 0)

		with open('./labels/%s_hsv.pkl' %(item_name), 'wb') as f:
			pickle.dump(info_hsv[1:], f)


		# YCR_CB color space		
		img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)		
		
		fea_ycrcb_0 = img_ycrcb[:,:,0]
		fea_ycrcb_1 = img_ycrcb[:,:,1]
		fea_ycrcb_2 = img_ycrcb[:,:,2]

		F_ycrcb_0 = fea_ycrcb_0[point_interest]
		F_ycrcb_1 = fea_ycrcb_1[point_interest]
		F_ycrcb_2 = fea_ycrcb_2[point_interest]

		temp_ycrcb = np.column_stack((F_ycrcb_0, F_ycrcb_1, F_ycrcb_2))
		info_ycrcb = np.concatenate((info_ycrcb, temp_ycrcb), axis = 0)

		with open('./labels/%s_ycrcb.pkl' %(item_name), 'wb') as f:
			pickle.dump(info_ycrcb[1:], f)


		# RGB color space			
		fea_rgb_0 = img_rgb[:,:,0]
		fea_rgb_1 = img_rgb[:,:,1]
		fea_rgb_2 = img_rgb[:,:,2]

		F_rgb_0 = fea_rgb_0[point_interest]
		F_rgb_1 = fea_rgb_1[point_interest]
		F_rgb_2 = fea_rgb_2[point_interest]

		temp_rgb = np.column_stack((F_rgb_0, F_rgb_1, F_rgb_2))
		info_rgb = np.concatenate((info_rgb, temp_rgb), axis = 0)

		with open('./labels/%s_rgb.pkl' %(item_name), 'wb') as f:
			pickle.dump(info_rgb[1:], f)

		'''for color in color_space:
			if color == 'rgb':
				img_fea = img_rgb
			elif color == 'ycrcb':
				img_fea = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
			else:
				img_fea = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
		
			fea_0 = img_fea[:,:,0]
			fea_1 = img_fea[:,:,1]
			fea_2 = img_fea[:,:,2]

			F_0 = fea_0[point_interest]
			F_1 = fea_1[point_interest]
			F_2 = fea_2[point_interest]

			temp = np.column_stack((F_0,F_1,F_2))

			info = np.concatenate((info, temp), axis = 0)
			#print(info[1:])
		

		
			# store label coordinates in labels dictionary
			#labels[str(im_file)] = (Roi1.x, Roi1.y)

			#print(np.shape(info))

			with open('../labels/%s_%s.pkl' %(item_name, color), 'wb') as f:
				pickle.dump(info[1:], f)'''

def main():
	label(args.item)

if __name__ == '__main__':
	main()




