import cv2
import os.path
import numpy as np
import random
import pathlib
import sys
import math


class Annotation:
    def __init__(self, label, cx, cy, w, h):
        self.label = label
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.snr = None


def generateRandomCrop(cropHeight,cropWidth,frameHeight,frameWidth,annotationBB):
    # annotationBB = [label, cx ,cy , w ,h]
    cropCenter = [0,0]
    cropCenter[1]=np.floor(min(max(cropWidth/2, cropCenter[1]), frameWidth-cropWidth/2-1))
    cropCenter[0]=np.floor(min(max(cropHeight/2, cropCenter[0]), frameHeight-cropHeight/2-1))

    if annotationBB is not None: # not false positive
        label = annotationBB[0]
        cx = annotationBB[1]
        cy = annotationBB[2]
        w = annotationBB[3]
        h = annotationBB[4]

        partialObjectFactor = 1
        rowRandRange = [max(cropHeight/2, cy-cropHeight/2 + round(partialObjectFactor*h/2)),
                        min(frameHeight-cropHeight/2, cy+cropHeight/2 - round(partialObjectFactor*h/2))]
        colRandRange = [max(cropWidth/2, cx-cropWidth/2 + round(partialObjectFactor*w/2)),
                        min(frameWidth-cropWidth/2, cx+cropWidth/2 - round(partialObjectFactor*w/2))]
        if rowRandRange[0] <= rowRandRange[1]:
            cropCenter[0]=random.randint(rowRandRange[0],rowRandRange[1]+1)
        else:
            print('object larger than crop')
            cropCenter[0]=round(np.mean(rowRandRange))
        if colRandRange[0] <= colRandRange[1]:
            cropCenter[1]=random.randint(colRandRange[0],colRandRange[1]+1)
        else:
            print('object larger than crop')
            cropCenter[1]=round(np.mean(colRandRange))

        newCenter=[np.floor(cy-(cropCenter[0]-cropHeight/2)), np.floor(cx-(cropCenter[1]-cropWidth/2))]

        newBBtmp=[label, newCenter[1]/cropWidth, newCenter[0]/cropHeight, w/cropWidth, h/cropHeight] # in normalized center-x-y formalism
        # dealing with extreme cases - when object is larger then crop
        dxm=-min(newBBtmp[1]-newBBtmp[3]/2,0)
        dxp=max(newBBtmp[1]+newBBtmp[3]/2,1)-1
        dym=-min(newBBtmp[2]-newBBtmp[4]/2,0)
        dyp=max(newBBtmp[2]+newBBtmp[4]/2,1)-1
        # newBB = [label , newCx , newCy , newW , newH]
        newBB=[label, (newCenter[1]/cropWidth)+dxm/2-dxp/2, (newCenter[0]/cropHeight)+dym/2-dyp/2, (w/cropWidth)-dxm-dxp, (h/cropHeight)-dym-dyp] # in normalized center-x-y formalism
    else:
        newBB = None
    # cropCoor = [topRow,bottomRow, leftColumn , rightColumn]
    cropCoor = [int(cropCenter[0]-cropHeight/2),int(cropCenter[0]+cropHeight/2), int(cropCenter[1]-cropWidth/2),int(cropCenter[1]+cropWidth/2)]
    
    return newBB,cropCoor
   

def read_frame(start, filename, height, width, bgr=False):
    # returns the frame in position 'start' from file 'filename'
    # if bgr is True - the function returns the frame in bgr (uint8) format

    frame_size = width * height
    _, extension = os.path.splitext(filename)

    if extension == ".raw2":
        with open(filename, 'rb') as f:
            f.seek(start*height*width*2, os.SEEK_SET)
            current_frame = np.fromfile(f, dtype=np.int16, count=height*width)
            current_frame = current_frame.reshape((height, width))
            if bgr is True:
                current_frame = cv2.normalize(current_frame, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)
    elif extension == ".rawc":
        with open(filename, 'rb') as f:
            f.seek(int(start*height*width*1.5), os.SEEK_SET)
            current_frame = np.fromfile(f, dtype=np.uint8, count=int(frame_size*1.5), sep='')
            current_frame.shape = (int(height*1.5), width)
            if bgr is True:
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_YUV2BGR_I420)
    elif extension == ".yuv":
        with open(filename, 'rb') as f:
            current_frame = np.fromfile(f, count = frame_size*2 ,dtype=np.uint8)
            current_frame = current_frame.reshape(height, width, 2)
            if bgr is True:
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_YUV2BGR_I420)
    
    return current_frame


# converts a uint16 format nparray into uint8 format
def imagesc_gs(np_array):
    min_value = np.amin(np_array)
    max_value = np.amax(np_array)
    original_range = max_value - min_value
    # to avoid dividing by 0
    if original_range == 0:
        original_range = 65535

    scale = 255.0/np.float(original_range)
    transform_image = (np_array - min_value).astype(np.float) * scale
    transform_image = np.around(transform_image)
    transform_image = np.uint8(transform_image)

    return transform_image


def convertBoolToStr(boolStr):
    if boolStr == 'True':
        return True
    elif boolStr == 'False':
        return False


# overlays 'top_image' on top of 'base_image'
def overlay_images_in_position(base_image, top_image, position):
    base_image[position[1]:position[1] + top_image.shape[0], position[0]:position[0] + top_image.shape[1]] = top_image
    return base_image


def create_mosaic_image(frames_list, crop_size, rows, cols):
    mosaic_size = (crop_size[0]*rows, crop_size[1]*cols, 3)
    # base image:
    base_image = np.zeros(mosaic_size, np.uint8)
    # current frame position in the base mosaic:
    current_position = [0]*2

    for i in range(len(frames_list)):
        # current column position:
        current_position[0] = int(i % cols) * crop_size[0]
        # current row position:
        current_position[1] = int(i / rows) * crop_size[1]
            
        # overlay images:
        overlay_images_in_position(base_image, frames_list[i], current_position)

    return base_image


def find_start_end_point_rectangle(cx, cy, width, height):
    start_x = int(cx - (width/2))
    start_y = int(cy - (height/2))
    end_x = int(cx + (width/2))
    end_y = int((cy) + (height/2))

    return ((start_x, start_y), (end_x, end_y))


def get_crop_center_in_frame(cx, cy, frameWidth, frameHeight, cropW, cropH):
    crop_center=[0,0]
    has_upper_space = bool(cy - (cropH/2) >= 0)
    has_lower_space = bool(cy + (cropH/2) < frameHeight)
    has_right_space = bool(cx + (cropW/2) < frameWidth)
    has_left_space = bool(cx - (cropW/2) >= 0)
    new_cx = cx
    new_cy = cy

    if has_lower_space:
        if has_upper_space:
            crop_center[0] = cy
            new_cy = cropH/2
        else:
            crop_center[0] = int(cropH/2)
            
    else:
        crop_center[0] = frameHeight - (cropH/2) - 1
        new_cy = cy - (frameHeight - cropH)

    
    if has_left_space:
        if has_right_space:
            crop_center[1] = cx
            new_cx = cropW/2
        else:
            crop_center[1] = int(frameWidth - (cropW/2) - 1)
            new_cx = cx - (frameWidth - cropW)

    return (crop_center, new_cx, new_cy)


# calculates and returns the distance between p1 and p2
def calculate_distance_between_points(p1, p2):
    distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    return distance


def convert_str_label_to_int(label_str):
    label_int = None

    if label_str == 'drone':
        label_int = 0
    elif label_str == 'airplane':
        label_int = 1
    elif label_str == 'bird':
        label_int = 2
    elif label_str == 'UFO':
        label_int = 3
    elif label_str == 'airplane with lights':
        label_int = 4
    elif label_str == 'balloon':
        label_int = 5
    elif label_str == 'human':
        label_int = 6
    elif label_str == 'vehicle':
        label_int = 7
    elif label_str == 'drone with lights':
        label_int = 8
    elif label_str == 'single front light':
        label_int = 9
    
    return label_int


def extract_annotations_from_old_labeled_file(filename, num_frames):
    ann_file = open(filename, "r")
    num_lines = sum(1 for line in ann_file)
    ann_file.seek(0, os.SEEK_SET)
    
    annotations = [[] for i in range(num_frames)]

    line = ann_file.readline()
    for j in range(num_lines):
        if (len(line) >= 2):  # line is not empty
            elements = line.split(',')   #split line according to commas
            num_of_objs = int(elements[1])
            frame_num = int(elements[0])
            for i in range(1, num_of_objs + 1):
                cx = int(elements[2 + (i-1) * 5])
                cy = int(elements[3 + (i-1) * 5])
                w = int(elements[4 + (i-1) * 5])
                h = int(elements[5 + (i-1) * 5])
                if i == num_of_objs:
                    label = convert_str_label_to_int(elements[6 + (i-1) * 5][:-1])
                else:
                    label = convert_str_label_to_int(elements[6 + (i-1) * 5])  # returns label as id number
                
                if label is not None:
                    new_annotation = Annotation(label, cx, cy, w, h)
                    annotations[frame_num].append(new_annotation)
        line = ann_file.readline()
    
    ann_file.close()
    return annotations


def convert_bird_label_str_to_int(label):
    if label == 'u':
        return 3
    elif label == 'n':
        return 1
    elif label == 'b':
        return 2
    else:
        return 3


def extract_annotations_from_birds_format_file(filename, frames, num_frames):
    annotations = [[] for i in range(num_frames)]
    ann_file = open(filename, "r")
    line = ann_file.readline()
    result = line.find('.jpg')

    counter = 0
    while(result != -1):
        image_name = line
        frame_name = line[:-1]
        while(frame_name != frames[counter]):
            counter += 1
        line = ann_file.readline()
        if len(line) < 2:
            break
        result = line.find('.jpg')

        while(result == -1):
            if len(line) < 2:
                break
            elems = line.split(',')   #split line according to commas
            left_x = int(elems[0])
            top_y = int(elems[1])
            w = int(elems[2])
            h = int(elems[3])
            label = convert_bird_label_str_to_int(elems[4][:-1])

            cx = int(np.ceil(left_x + w/2))
            cy = int(np.ceil(top_y + h/2))

            new_annotation = Annotation(label, cx, cy, w, h)
            annotations[counter].append(new_annotation)

            line = ann_file.readline()
            result = line.find('.jpg')

        counter += 1
        print(counter)
    
    ann_file.close()

    return annotations
                    

def extract_lines_from_file(filename):
    lines = []
    with open(filename) as f:
        for line in f:
            lines.append(line)
    return lines


def read_binary_map(map, index, width, height, normalize=True):
    frame_size = width * height

    with open(map, 'rb') as f:
        f.seek(index * frame_size * 4, os.SEEK_SET)
        frame = np.fromfile(f, dtype=np.float32, count=frame_size)
        frame = frame.reshape((height, width))
    
    # frame_flattened = frame.flatten()
    # max_value = max(frame_flattened)
    # print(str(max_value))
    # resized_width = width * 2
    # resized_height = height * 2
    # frame = cv2.resize(frame, (resized_width, resized_height))
    if normalize is True:
        cv2.normalize(src=frame, dst=frame, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX)

    return frame
    

def get_number_of_images_in_folder(folder):
    counter = 0
    for file in os.listdir(folder):
        if file.endswith(".bmp") or file.endswith(".jpg") or file.endswith(".png") or file.endswith(".yuv"):
            counter += 1
    
    return counter


def get_number_of_frames_in_binary_file(filename, frame_bytes_size):
    array_length = os.stat(filename).st_size
    number_of_frames = int(array_length/frame_bytes_size)

    return number_of_frames


def adjust_frames_size(frame1, frame2):
    frame1_width = frame1.shape[1]
    frame1_height = frame1.shape[0]
    frame2_width = frame2.shape[1]
    frame2_height = frame2.shape[0]

    if frame1_width < frame2_width and frame1_height < frame2_height:
        frame2 = cv2.resize(frame2, (frame1_width, frame1_height), interpolation = cv2.INTER_AREA)
        frame2.shape = (frame1_height, frame1_width, 3)
    
    elif frame1_width > frame2_width and frame1_height > frame2_height:
        frame1 = cv2.resize(frame1, (frame2_width, frame2_height), interpolation = cv2.INTER_AREA)
        frame1.shape = (frame2_height, frame2_width, 3)
    
    return (frame1, frame2)


def extract_detections_from_gt_file(gt_filename, frame_width=None, frame_height=None, yolo=False):
    annotations = []
    current_gt = open(gt_filename, "r")
    gt_lines = current_gt.readlines()
    for line in gt_lines:
        if(len(line) >= 2):
            elements = line.split(' ')
            label = int(elements[0])
            cx = float(elements[1])
            cy = float(elements[2])
            w = float(elements[3])
            h = float(elements[4])
            if yolo is False:
                cx = int(cx * frame_width)
                cy = int(cy * frame_height)
                w = int(w * frame_width)
                h = int(h * frame_height)
            annotation = Annotation(label, cx, cy, w, h)
            annotations.append(annotation)
    
    current_gt.close()
    return annotations


def extract_annotations(gt_folder, frame_width=None, frame_height=None, yolo=False):
    gt_filenames = []
    for file in os.listdir(gt_folder):
        if file.endswith(".txt"):
            gt_filenames.append(os.path.join(gt_folder, file))
    
    gt_filenames.sort()
    num_gts = len(gt_filenames)

    gt_files = [[] for i in range(num_gts)]

    for i in range(num_gts):
        gt_files[i] = extract_detections_from_gt_file(gt_filenames[i], frame_width, frame_height, yolo)

    return gt_files


def frame_count(video_path, manual=False):
    def manual_count(handler):
        frames = 0
        while True:
            status, frame = handler.read()
            if not status:
                break
            frames += 1
        return frames 

    cap = cv2.VideoCapture(video_path)
    # Slow, inefficient but 100% accurate method 
    if manual:
        frames = manual_count(cap)
    # Fast, efficient but inaccurate method
    else:
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except:
            frames = manual_count(cap)
    cap.release()
    
    return frames