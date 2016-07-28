import cv2
import multiprocessing as mp
import math
import numpy as np
import datamani
import drMatches
from drMatches import Position
import FrameProcessing
from FrameProcessing import processImage
import time

def getFrame(queue, startFrame, endFrame, i, videoFile, frameCounts, indices, tails, fps, img, data):
    cap = cv2.VideoCapture(videoFile)
    for frame in range(startFrame, endFrame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        print 'Current frame: '+ str(frame)         
        frameNo = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, f = cap.read()
        f = processImage(f, frame, i, fps, frameCounts, indices, tails, img, data)
        if ret:
            # try:
            #     queue.put([frameNo, f])
            # except:
            #     queue.append([frameNo, f])
            queue.append([frameNo, f])
    cap.release()

def singleProcess(processCount, fileLength, videoFile, fps, img, data):
    frameQueue = []

    bunches, frameCounts, indices, tails = createArrays(1, fileLength, fps)

    getFrame(frameQueue, 0, fileLength - 1, 0, videoFile, frameCounts, indices, tails, fps, img, data)

    results = []

    for i in range(bunches[0][0], bunches[0][1] - 1):
        results.append(frameQueue[i])

    return results, False, None, None

def multiProcess(processCount, fileLength, videoFile, fps, img, data):
    qList = []
    for i in range(processCount):
    	qList.append(mp.JoinableQueue())

    bunches, frameCounts, indices, tails = createArrays(processCount, fileLength, fps)

    getFrames = []
    for i in range(processCount):
        getFrames.append(mp.Process(target=getFrame, args=(qList[i], bunches[i][0], bunches[i][1], i, videoFile, frameCounts, indices, tails, fps, img, data)))
    # print "FrameCount:"+'             '+ "Low Time"+'              '+"Actual Time: "+'          '+ "High Time "

    for process in getFrames:
        process.start()

    results = []
    for i in range(len(qList)):
        results.append([qList[i].get() for p in range(bunches[i][0], bunches[i][1])])

    # terminate(getFrames, qList)
    return results, True, getFrames, qList
def divideFrames(processCount, fileLength):
    bunches = []
    ratio = int(fileLength/processCount)
    for startFrame in range(0, fileLength, ratio):
        endFrame = startFrame + ratio
        if fileLength-startFrame< 2*ratio:
            endFrame = fileLength
            bunches.append((startFrame, endFrame))
            break
        bunches.append((startFrame, endFrame))
    return bunches

def createArrays(processCount, fileLength, fps):
    FRAMECOUNT, IDX, TAIL = [], [], []
    bunches = divideFrames(processCount, fileLength)
    for i in range(len(bunches)):
        # print i
        FRAMECOUNT.append(bunches[i][0] * 1000.0/fps)
        IDX.append(bunches[i][0])
        TAIL.append(bunches[i][0]+10)
    return bunches, FRAMECOUNT, IDX, TAIL

def terminate(processes, queues):
    for process in processes:
        process.terminate()
        process.join()

    for queue in queues:
        queue.close()