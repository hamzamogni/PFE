#! /usr/bin/python3.6

import os
from shutil import copy2

source_dir = "data/data_orijinal"
target_dir = "data_preprocessed"

percentage_train = 80
percentage_test  = 100 - percentage_train

try:
	os.mkdir(target_dir)
	os.mkdir(target_dir + "/train")
	os.mkdir(target_dir + "/test")
except FileExistsError as e:
	pass



for directory in os.listdir(source_dir):
	class_dir = os.path.join(source_dir, directory)
	nbr_class_samples = len(
		[
			name for name in os.listdir(class_dir) 
			if os.path.isfile(os.path.join(class_dir, name))
		]
	)

	nbr_train = round(nbr_class_samples * percentage_train / 100)

	try:
		os.mkdir(os.path.join(target_dir, "test", directory))
		os.mkdir(os.path.join(target_dir, "train", directory))
	except FileExistsError as e:
		pass

	images = os.listdir(class_dir)

	to_train = images[:nbr_train]
	to_test  = images[nbr_train:]
	
	
	for image in to_train:
		copy2(
			os.path.join(class_dir, image),
			os.path.join(target_dir, "train", directory, image)
		)


	for image in to_test:
		copy2(
			os.path.join(class_dir, image),
			os.path.join(target_dir, "test", directory, image)
		)
	
	print("split class " + directory)

