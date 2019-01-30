import os, json, pickle, gzip, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import model
from model import Single_Gaussian


folder = './validset'
result_folder = './result'

####{barrel_blue, not_barrel_blue, brown, green}
def load_fea_data(name, color):
	with open('./labels/%s_%s.pkl' %(name, color), 'rb') as f:
		tra = pickle.load(f)
	return tra

def load_image(name, color):
	img_bgr = cv2.imread(os.path.join(folder, name))
	#img_bgr = np.uint8(np.clip((1.5*img_bgr+10), 0, 255))
	if color == 'rgb':
		out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
	elif color == 'hsv':
		out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	else:
		out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
	pixels = np.reshape(out_img, (-1,3))
	if color == 'hsv':
		pixels = pixels[:,:2]
	#print(pixels.shape)
	#print(type(img_rgb))
	return pixels
	

def main():
	im_size = [800,1200]
	num_pixel = im_size[0]*im_size[1]
	#out_im = np.zeros(im_size)
	color_space = ['rgb', 'ycrcb', 'hsv']
	#model_name = 'single_gaussian'
	#test_name = '../validset/38.png'
	kernel = np.ones((3,3), np.uint8)
	model_parameter = []
	
	for color in color_space:
		# load testing image and store image name
		if color == 'hsv':
			test = np.zeros((1, 2))
		else:
			test = np.zeros((1,3))
		test_fname = []
		for test_file in os.listdir(folder):
			test_fname.append(test_file)
			temp_test = load_image(test_file, color)
			test = np.concatenate((test, temp_test), axis = 0)

		test = test[1:]
			
		num_test = len(test_fname)


		bb_tra = load_fea_data('barrel_blue', color)
		nbb_tra = load_fea_data('not_barrel_blue', color)
		bro_tra = load_fea_data('brown', color)
		gre_tra = load_fea_data('green', color)
		
		if color == 'hsv':
			test = test[:, :2]
			bb_tra = bb_tra[:, :2]
			nbb_tra = nbb_tra[:, :2]
			bro_tra = bro_tra[:, :2]
			gre_tra = gre_tra[:, :2]

		model_bb = Single_Gaussian(bb_tra)
		model_nbb = Single_Gaussian(nbb_tra)
		model_bro = Single_Gaussian(bro_tra)
		model_gre = Single_Gaussian(gre_tra)

		# show Gaussian model parameter
		#print('color = ', color)
		model_parameter.append(model_bb.show_mean_cov())
		model_parameter.append(model_nbb.show_mean_cov())
		model_parameter.append(model_bro.show_mean_cov())
		model_parameter.append(model_gre.show_mean_cov())
		
		ans_bb = model_bb.predict(test)
		ans_nbb = model_nbb.predict(test)
		ans_bro = model_bro.predict(test)
		ans_gre = model_gre.predict(test)		

		#ans_bb = model.single_gaussian(test, bb_tra)
		#ans_nbb = model.single_gaussian(test, nbb_tra)
		#ans_bro = model.single_gaussian(test, bro_tra)
		#ans_gre = model.single_gaussian(test, gre_tra)

		# find max
		tmp1 = ans_bb.tolist()
		tmp2 = ans_nbb.tolist()
		tmp3 = ans_bro.tolist()
		tmp4 = ans_gre.tolist()
		max_tmp = list(map(max, zip(tmp1, tmp2, tmp3, tmp4)))
		max_tmp = np.array(max_tmp)

		point_predict = 255*np.array(max_tmp == ans_bb).astype('uint8')
		#print(np.shape(point_predict))
		for i in range(num_test):
			#print(np.shape(point_predict[i*num_pixel:(i+1)*num_pixel,:]))
			#print(i)
			test_predict = point_predict[i*num_pixel:(i+1)*num_pixel].reshape(im_size)
			seg_name = './seg/%s%s.png' %(test_fname[i][:-4], color)
			cv2.imwrite(seg_name, test_predict)
			# optimize segmentation
			img_optimize = test_predict
			iter_num = 3
			while True:
				img_optimize = cv2.erode(img_optimize, kernel, iterations = iter_num)
				img_optimize = cv2.dilate(img_optimize, kernel, iterations = iter_num)

				contours, hierarchy = cv2.findContours(img_optimize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
				if np.shape(contours)[0] <= 2:
					break
				else:
					iter_num = iter_num + 2

			print('finish optimization %s' %(test_fname[i]))

			# Draw contour
			img = cv2.imread(os.path.join(folder, test_fname[i]))
			for contour in contours:
				rect = cv2.minAreaRect(contour)
				box = np.int0(cv2.boxPoints(rect))
				cv2.drawContours(img,[box],0,(0, 0, 255),2)
			
			# store result image
			out_file_name = '%s/%s%s.png' %(result_folder, test_fname[i][:-4], color)
			cv2.imwrite(out_file_name, img)		
	

	with open('model_parameter.pkl', 'wb') as f:
		pickle.dump(model_parameter, f)	
	'''	
	###load feature(dimension = ...*3     3 for channel(RGB))
	bb_tra = load_fea_data('barrel_blue')
	nbb_tra = load_fea_data('not_barrel_blue')
	bro_tra = load_fea_data('brown')
	gre_tra = load_fea_data('green')
	#
	val = load_image(test_name)
	ans_bb = model.single_gaussian(val, bb_tra)
	ans_nbb = model.single_gaussian(val, nbb_tra)
	ans_bro = model.single_gaussian(val, bro_tra)
	ans_gre = model.single_gaussian(val, gre_tra)
	#print(np.shape(ans_bb))
	# find max
	tmp1 = ans_bb.tolist()
	tmp2 = ans_nbb.tolist()
	tmp3 = ans_bro.tolist()
	tmp4 = ans_gre.tolist()
	max_tmp = list(map(max, zip(tmp1, tmp2, tmp3, tmp4)))
	max_tmp = np.array(max_tmp)
	

	img = cv2.imread(os.path.join(test_name))
	point_predict = 255*np.array(max_tmp == ans_bb).reshape(im_size).astype('uint8')
	#print(point_predict)
	kernel = np.ones((3,3), np.uint8)
	
	# optimize segmentation
	img_optimize = point_predict
	iter_num = 3
	while True:
		img_optimize = cv2.erode(img_optimize, kernel, iterations = iter_num)
		img_optimize = cv2.dilate(img_optimize, kernel, iterations = iter_num)
		#cv2.imshow('segmentation', img_optimize)
		contours, hierarchy = cv2.findContours(img_optimize, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#print(np.shape(contours)[0])
		if np.shape(contours)[0] <= 2:
			break
		else:
			iter_num = iter_num + 2
	#print(np.shape(contours))
	print('finish optimization')
	for contour in contours:
		rect = cv2.minAreaRect(contour)
		box = np.int0(cv2.boxPoints(rect))
		cv2.drawContours(img,[box],0,(0, 0, 255),2)
		#cv2.imshow('img', img)
		#cv2.waitKey(0)
		
	#cv2.drawContours(img, contours, -1, (0,0,255),3)
	cv2.imshow('img', img)
	cv2.waitKey(0)
	#img_gray = cv2.cvtColor(point_predict, cv2.COLOR_RGB2GRAY)
	#ret, thresh = cv2.threshold(point_predict, 0, 255, 0)

	#plt.imshow(point_predict)
	#plt.show()

	#ans_nbb = model.single_gaussian(nbb_val, nbb_tra)	
	#ans_bro = model.single_gaussian(bro_val, bro_tra)
	#ans_gre = model.single_gaussian(gre_val, gre_tra)
	'''

if __name__ == '__main__':
	main()
