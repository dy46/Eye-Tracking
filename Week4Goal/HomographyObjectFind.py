import cv2
import multiprocessing as mp
import math
import numpy as np
import datamani
import drMatches
from drMatches import Position
import time
import sys

def edit(f, currentFrame, i):
    global IDX
    global TAIL
    global FRAMECOUNT
    framecount = currentFrame * 1000.0/fps
    idx = IDX[i]
    tail = TAIL[i]
    img3 = 0
    img2 = f ## get frame from the queue
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    s=np.zeros((4,4))
    #print s.dtype
    First = True
    pos = Position(0,0,0,0)
    flag = True
    for i,img1 in enumerate(img):
        ignore = False;
        if First == False:
            img2 = img3
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,pos.getColor(),3, cv2.LINE_AA)
            # cv2.circle(img2, (int(float(PoRBX[idx])),int(float(PoRBY[idx]))), 10, (255, 0, 0), 20)
        else:
            # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None
            ignore = True

        if First == True:
            img2, x, y, idx, tail ,ignore= datamani.drawCircle(img2, framecount, videoData, idx, tail, fps, ignore)
            framecount = framecount + (1.0/fps)*1000.0
        img3, pos = drMatches.drawMatches(img1,kp1,img2,kp2,good, pos) ## line must not execute
        First = False
        # print str(mp.current_process()) + "Ignore: " + str(ignore)
        if not ignore:
            s[i][0]=dst[0][0][0];
            s[i][1]=dst[3][0][0];
            s[i][2]=dst[0][0][1];
            s[i][3]=dst[2][0][1];
            if x>s[i][0] and x<s[i][1] and y>s[i][2] and y<s[i][3]:
                cv2.putText(img3,'Gazing at the '+str(i+1)+' object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
                flag = False
        ### HERE WHAT I WANT TO DO IS PROCESS THIS WITH A DIFFERENT IMAGE USING IMG3 as my new framw
    if flag:
        cv2.putText(img3,'Gazing at none of the object',(250,30), font, 1,(255,255,255),2,cv2.LINE_AA)
    IDX[i] = idx
    TAIL[i] = tail
    FRAMECOUNT[i] = framecount
    return img3

def getFrame(queue, startFrame, endFrame, fourcc, fps, capSize, i):
    cap = cv2.VideoCapture(file)  # crashes here
    # print("opened capture {}".format(mp.current_process()))
    for frame in range(startFrame, endFrame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)  # opencv3
        # print frame            
        frameNo = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # opencv3
        ret, f = cap.read()
        f = edit(f, frame, i)
        if ret:
            # print("{} - put ({})".format(mp.current_process(), frameNo))
            queue.put((frameNo, f))
    cap.release()


if __name__ == '__main__':
    ######## Initialize Constants ########
    MIN_MATCH_COUNT = 5
    # framecount = 0.0;
    i = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    FRAMECOUNT = []
    IDX = []
    TAIL = []
    img = [cv2.imread('1.png',0), cv2.imread('feature1.png', 0)] ## Reads in comparison images
    videoData = datamani.createVideoData(open('1.txt', 'r')) ## Reads in data file
    file = "cuttwo.mp4"
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
    success = cv2.VideoWriter('Perrovida.mp4',fourcc,fps,capSize)
    # get cpuCount for processCount
    # processCount = mp.cpu_count() / 3
    processCount = 3
    qList = []
    for i in range(processCount):
        qList.append(mp.JoinableQueue())
    # inQ1 = mp.JoinableQueue()  # not sure if this is right queue type, but I also tried mp.Queue()
    # inQ2 = mp.JoinableQueue()
    # inQ3 = mp.JoinableQueue()
    # qList = [inQ1, inQ2, inQ3]
    # print fileLen
    # set up bunches
    bunches = []
    ratio = int(fileLen/processCount)
    for startFrame in range(0, fileLen, ratio):
        endFrame = startFrame + ratio
        if fileLen-startFrame< 2*ratio:
            endFrame = fileLen
            bunches.append((startFrame, endFrame))
            break
        bunches.append((startFrame, endFrame))
    # print bunches
    getFrames = []
    for i in range(processCount):
        getFrames.append(mp.Process(target=getFrame, args=(qList[i], bunches[i][0], bunches[i][1], fourcc, fps, capSize, i)))
    # print "FrameCount:"+'             '+ "Low Time"+'              '+"Actual Time: "+'          '+ "High Time "

    for i in range(len(bunches)):
        # print i
        FRAMECOUNT.append(bunches[i][0] * 1000.0/fps)
        IDX.append(bunches[i][0])
        TAIL.append(bunches[i][0]+10)

    for process in getFrames:
        process.start()

    results = []
    for i in range(len(qList)):
        results.append([qList[i].get() for p in range(bunches[i][0], bunches[i][1])])

    # results1 = results[0]
    # results2 = results[1]
    # results3 = results[2]
    # results1 = [inQ1.get() for p in range(bunches[0][0], bunches[0][1])]
    # results2 = [inQ2.get() for p in range(bunches[1][0], bunches[1][1])]
    # results3 = [inQ3.get() for p in range(bunches[2][0], bunches[2][1])]

    for result in results:
        for i, frame in enumerate(result):
            # if frame.size == 0:
            #     print 'Well it looks like there is an empty image. '+'Frame: '+str(i)
            #     sys.exit()
            success.write(frame[1])

    # for i in results1:
    #     success.write(i[1])
    # for i in results2:
    #     success.write(i[1])
    # for i in results3:
    #     success.write(i[1])

    for process in getFrames:
        process.terminate()
        process.join()

    for queue in qList:
        queue.close()

    success.release()


