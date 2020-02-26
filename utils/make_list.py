import os
import pandas as pd
from sklearn.utils import shuffle

image_list = []
label_list = []
#your data file path
image_dir = r'E:/data/ColorImage/'
label_dir = r'E:/data/Label/'

"""
   ColorImage/
     road02/
       Record002/
         Camera 5/
           ...
         Camera 6
       Record003
       ....
     road03
     road04
   Label/
     Label_road02/
      Label
       Record002/
         Camera 5/
          ...
         Camera 6
       Record003
       ....
     Label_road03
     Label_road04     
     
"""
for s1 in os.listdir(image_dir):
	image_sub_dir1 = os.path.join(image_dir, s1)
	label_sub_dir1 = os.path.join(label_dir, 'Label_'+s1, 'Label')
	for s2 in os.listdir(image_sub_dir1):
		image_sub_dir2 = os.path.join(image_sub_dir1, s2)
		label_sub_dir2 = os.path.join(label_sub_dir1, s2)
		for s3 in os.listdir(image_sub_dir2):
			image_sub_dir3 = os.path.join(image_sub_dir2, s3)
			label_sub_dir3 = os.path.join(label_sub_dir2, s3)
			for s4 in os.listdir(image_sub_dir3):
				s4l = s4.replace('.jpg', '_bin.png')
				image_sub_dir4 = os.path.join(image_sub_dir3, s4)
				label_sub_dir4 = os.path.join(label_sub_dir3, s4l)
				if not os.path.exists(image_sub_dir4):
					print(image_sub_dir4)
					continue
				if not os.path.exists(label_sub_dir4):
					print(label_sub_dir4)
					continue
				image_list.append(image_sub_dir4)
				label_list.append(label_sub_dir4)

total_length = len(image_list)
assert total_length == len(label_list)
print("The length of image dataset is {}, and label is {}".format(len(image_list), len(label_list)))
sixth_length = int(total_length*0.6)
eight_length = int(total_length*0.8)

data = pd.DataFrame({'image': image_list, 'label': label_list})
data_shuffle = shuffle(data)

train_dataset = data_shuffle[: sixth_length]
val_dataset = data_shuffle[sixth_length: eight_length]
test_dataset = data_shuffle[eight_length:]

train_dataset.to_csv('../data_list/train.csv', index=False)
val_dataset.to_csv('../data_list/val.csv', index=False)
test_dataset.to_csv('../data_list/test.csv', index=False)