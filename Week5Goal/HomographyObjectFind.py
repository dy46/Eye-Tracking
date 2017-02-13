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
import gui

def writeFrames(result, success):
    for frame in result:
        if len(frame) == 0:
            print 'Well it looks like there is an empty image. '+'Frame: '+str(i)
        success.write(frame[1])

if __name__ == '__main__':
    ######## Initialize Constants ########
    # framecount = 0.0;
    i = 0 ## Reads in comparison images
    file="Three_Objects_No_Point_Short.mp4"
    multi_flag=False
    refimg,file,multi_flag=gui.start()
    print refimg, file, multi_flag
    img=[]
    for i in refimg:
        img.append(cv2.imread(i, 0))
    videoData = datamani.createVideoData(open('Three_Objects_Raw_Data.txt', 'r')) ## Reads in data file

    capture_temp = cv2.VideoCapture(file)
    fileLen = int((capture_temp).get(cv2.CAP_PROP_FRAME_COUNT))  # opencv3
    fps = capture_temp.get(cv2.CAP_PROP_FPS) ##fps
    # print "fps"+str(fps)
    ret,temp=capture_temp.read(); ## Reads the first frame
    capture_temp.release()
    height, width = temp.shape[:2]
    # print "height"+str(height)
    # print "width"+str(width)
    # print "File length: "+ str(fileLen)
    capSize = (width,height) ## this is the size of my source video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') ## starts ouput file
    success = cv2.VideoWriter('Please_Work.mp4',fourcc,fps,capSize)
    # get cpuCount for processCount
    # processCount = mp.cpu_count() / 3
    # print "Processing"
    processCount = 4
    if multi_flag:
        results, multi_flag, getFrames, qList = multiProcess(processCount, fileLen, file, fps, img, videoData)
    else:
        results, multi_flag, getFrames, qList = singleProcess(processCount, fileLen, file, fps, img, videoData)

    #single
    #multi
    # results1 = results[0]
    # results2 = results[1]
    # results3 = results[2]
    # results1 = [inQ1.get() for p in range(bunches[0][0], bunches[0][1])]
    # results2 = [inQ2.get() for p in range(bunches[1][0], bunches[1][1])]
    # results3 = [inQ3.get() for p in range(bunches[2][0], bunches[2][1])]

    # print "Done"

    if multi_flag:
        for result in results:
            writeFrames(result, success)
    else:
        writeFrames(results, success)

    # for i in results1:
    #     success.write(i[1])
    # for i in results2:
    #     success.write(i[1])
    # for i in results3:
    #     success.write(i[1])
    if multi_flag:
        processing.terminate(getFrames, qList)
    success.release()
    print 'I am fucken DONE'


