import cv2
import numpy as np
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--video", help="Path to video file to label",
                    action="store")
parser.add_argument("--labels", help="Path to label file to use",
                    action="store")
parser.add_argument("--fname", help="Output folder name",
                    action="store", default='output')
args = parser.parse_args()

video_path = args.video
labels_path = args.labels
output_path = args.fname

def get_label_duration(labels, label_index, FPS):
  label = ''.join(str(e) for e in labels[label_index]['left']) + ''.join(str(e) for e in labels[label_index]['right'])
  duration = labels[label_index]['to'] * FPS
  return label, duration

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(video_path)
 
# Get video framerate
FPS = cap.get(cv2.CAP_PROP_FPS)

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
labels = []
with open('labels.json') as json_file:
    labels = json.load(json_file)['actions']

# Keep track of which label we are using
label_cnt = 0
# Start from first label
label, duration = get_label_duration(labels, label_cnt, FPS)

# Read until video is completed
frame_cnt = 0
while(cap.isOpened()):
  # Read video frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # Check if we have reached the timespot for the next label
    # If duration is negative, then that must have been the last label
    if frame_cnt >= duration and duration > 0:
      label_cnt += 1
      label, duration = get_label_duration(labels, label_cnt, FPS)

    # TODO perform image augmentation here
    
    cv2.imwrite(output_path + '/frame_' + str(frame_cnt) + '_' + label + '.jpg', frame)
    frame_cnt += 1
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()