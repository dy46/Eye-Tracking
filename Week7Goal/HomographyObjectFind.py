import cv2
import multiprocessing as mp
import math
import numpy as np
import datamani
import drMatches
from drMatches import Position
import time
import processing
from processing import multiProcess, singleProcess

def writeFrames(result, success):
    for frame in result:
        if len(frame) == 0:
            print 'Well it looks like there is an empty image. '+'Frame: '+str(i)
        success.write(frame[1])

if __name__ == '__main__':
    ######## Initialize Constants ########
    i = 0
    video_file = "madison_david_cut.mp4"
    process_count = 4
    img = [cv2.imread('frosted_flakes.png',0)] ## Reads in comparison images
    video_data = datamani.createVideoData(open('mady.txt', 'r')) ## Reads in data file
    capture_temp = cv2.VideoCapture(video_file)
    file_length = int((capture_temp).get(cv2.CAP_PROP_FRAME_COUNT))  # opencv3
    fps = capture_temp.get(cv2.CAP_PROP_FPS) ##fps
    ret, temp = capture_temp.read(); ## Reads the first frame
    capture_temp.release()
    height, width = temp.shape[:2]
    capture_size = (width,height) ## this is the size of my source video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') ## starts ouput file
    success = cv2.VideoWriter('Please_Work_7.mp4',fourcc,fps,capture_size)
    results, multi_flag, get_frames, qList = singleProcess(process_count, file_length, video_file, fps, img, video_data)
    if multi_flag:
        for result in results:
            writeFrames(result, success)
    else:
        writeFrames(results, success)
    if multi_flag:
        processing.terminate(get_frames, qList)
    success.release()
    print 'Finished'


