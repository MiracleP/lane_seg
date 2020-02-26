from utils.data_preprocess import resize_color_data
from PIL import Image
import os
import cv2

def resize_img():
    dir = r'predict'
    i = 0
    path = 'predict/resize_data'
    if not os.path.exists(path):
        os.makedirs(path)
    for name in os.listdir(dir):
        img = Image.open(dir+'/'+ name)
        new_img = resize_color_data(img)
        new_img.save(path+'/lane_data{}.png'.format(i))
        i += 1

def img2video(path):
    fps = 10
    size = (3384, 1710)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter('bd.avi', fourcc, fps, size)
    for name in os.listdir(path):
        frame = cv2.imread(path+'/'+name)
        video_writer.write(frame)
    video_writer.release()

def main():
    path = 'E:/data/ColorImage/road02/Record001/Camera 5'
    img2video(path)

if __name__ == '__main__':
    main()

