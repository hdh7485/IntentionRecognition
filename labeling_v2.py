import cv2
import numpy as np
import json
import argparse
import os
from imgaug import augmenters as iaa
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="Path to video file to label", default="videos",
                    action="store")
parser.add_argument("--labels", help="Path to label file to use", default="json",
                    action="store")
parser.add_argument("--fname", help="Output folder name",
                    action="store", default='output')
args = parser.parse_args()

video_path = args.video
labels_path = args.labels
output_path = args.fname

def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for fi in f:
            files.append(os.path.join(r, fi))
    return files

def time_count(time):
    time = time.split(':')
    return int(time[0])*60 +int(time[1])


def get_label_duration(labels, label_index, FPS):
    label = ''.join(str(e) for e in labels[label_index]['left']) + ''.join(str(e) for e in labels[label_index]['right'])
    duration = (time_count(labels[label_index]['to'])-time_count(labels[label_index]['from'])) * FPS
    return label, duration


def split_and_aug(video_path, json_path):
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(str(video_path))
    # Get video framerate
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    FPS = cap.get(cv2.CAP_PROP_FPS)
     
    labels = []
    label_list = []
    with open(str(json_path)) as json_file:
        labels = json.load(json_file)['actions']
    
    frame_end = time_count(labels[len(labels)-1]['to'])* FPS
    
    print(FPS)
    print(frame_end)
    # Keep track of which label we are using
    label_cnt = 0
    # Start from first label
    label, duration = get_label_duration(labels, label_cnt, FPS)
    print('label_cnt ={0} label is {1} duration is {2}'.format(label_cnt, label, duration))
    
    # Read until video is completed
    frame_cnt = 0
    cnt = 0
    c = 0
    while(cap.isOpened()):
        # Read video frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (224,224))
        if ret == True and frame_cnt <= frame_end :
            # Check if we have reached the timespot for the next label
            # If duration is negative, then that must have been the last label
            if frame_cnt >= duration and duration > 0 and label_cnt < len(labels):
                print('label_cnt ={0} label is {1} duration is {2}'.format(label_cnt, label, duration))
                label, duration = get_label_duration(labels, label_cnt, FPS)
                b = range(int(duration))
                c = c + duration
                print(c)
                for i in b:
                    label_list.append(label)
                label_cnt += 1
            # Perform image augmentation, then saved all augmented images in different folders
            augments = [iaa.Affine(rotate=(-25, 25)), iaa.GaussianBlur(0.5), iaa.LinearContrast(1.2), iaa.AdditiveGaussianNoise(scale=0.05*100)]
            names = ['affine', 'blur', 'linearcontrast', 'noise']
            results = [aug.augment_image(frame) for aug in augments]
    
            for img, name in zip(results, names): 
                path = output_path + '/' + name
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(path + '/' + os.path.basename(video_path) + str(frame_cnt) +'.jpg', img)
            if not os.path.exists(output_path + '/normal'):
                os.makedirs(output_path + '/normal')
            cv2.imwrite(output_path + '/normal/' + os.path.basename(video_path) + str(frame_cnt) + '.jpg', frame)
            #print('frame_cnt ={0} label is {1}'.format(frame_cnt, label))
            frame_cnt += 1
            # Break the loop
        else:  
          break

    label_list.append(0)
    df = pd.DataFrame(label_list)
    csv_path = output_path + '/csv'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    df.to_csv('output/csv/' + os.path.basename(video_path) + '.csv', index=False, header=False )
    print('finish')
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    video_files = get_files("videos")
    json_files = get_files("json")
    print(video_files)
    print(json_files)
    for video_file in video_files:
        json_file = video_file.replace("videos", "json").replace("avi", "json")
        print(json_file)
        split_and_aug(video_file, json_file)
