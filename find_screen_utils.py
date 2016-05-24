'''
Utilities to find monitor/screen coordinates within an image.
Used for eye tracking studies.
'''
import cv2
import numpy as np
import warnings
from sklearn import linear_model


def group_points(points):
    '''
    Groups set of points into edges of screen.
    '''
    horz = []
    vert = []
    prev = points[0]
    for pt in list(points):
        if np.array_equal(pt, prev):
            continue
        diff = np.abs(pt - prev)
        if diff[1] == 0:
            horz.append(pt)
        elif (diff[0]/diff[1]) > 5:
            horz.append(pt)
        elif diff[0] == 0:
            vert.append(pt)
        elif (diff[1]/diff[0]) > 5:
            vert.append(pt)
        prev = pt

    # find (very) approximate center of edges
    horz = np.array(horz)
    vert = np.array(vert)
    x_center = horz.mean(axis=0)[0]
    y_center = vert.mean(axis=0)[1]

    # split horizontal lines into top and bottom
    top = []
    bottom = []
    for x, y in list(horz):
        if y < y_center:
            top.append((x, y))
        else:
            bottom.append((x, y))
    top = np.array(top)
    bottom = np.array(bottom)

    # split vertical lines into left and right
    left = []
    right = []
    for x, y in vert:
        if x > x_center:
            right.append((x, y))
        else:
            left.append((x, y))
    left = np.array(left)
    right = np.array(right)

    return ((top, bottom), (left, right))


def group_lines(lines):
    '''
    Groups lines returned by HoughLinesP into top, bottom, left,
    and right edges.
    '''
    horz = []
    vert = []
    # split into horizontal and vertical lines
    for x1, y1, x2, y2 in lines:
        # if change in x is less than in y, then classify as vertical
        if np.abs(x1-x2) < np.abs(y1-y2):
            vert.append((x1, y1, x2, y2))
        else:
            horz.append((x1, y1, x2, y2))

    # find (very) approximate center of edges
    horz = np.array(horz)
    vert = np.array(vert)
    x_center = (horz.mean(axis=0)[0] + horz.mean(axis=0)[2]) / 2
    y_center = (vert.mean(axis=0)[1] + vert.mean(axis=0)[3]) / 2

    # split horizontal lines into top and bottom
    top = []
    bottom = []
    for x1, y1, x2, y2 in horz:
        if y1 < y_center:
            top.append((x1, y1))
            top.append((x2, y2))
        else:
            bottom.append((x1, y1))
            bottom.append((x2, y2))
    top = np.array(top)
    bottom = np.array(bottom)

    # split vertical lines into left and right
    left = []
    right = []
    for x1, y1, x2, y2 in vert:
        if x1 > x_center:
            right.append((x1, y1))
            right.append((x2, y2))
        else:
            left.append((x1, y1))
            left.append((x2, y2))
    left = np.array(left)
    right = np.array(right)

    def filter_outliers(group, axis):
        '''
        Remove outliers from the group along an axis.
        If axis=0, looking at x value (for vertical lines)
        If axis=1, looking at y value (for horz lines)
        '''
        filtered = []
        vals = group[:, axis]
        med = np.median(vals)
        resid = np.abs(vals - med)
        MAD = np.median(resid) * 1.4826
        for p in list(group):
            if abs(p[axis]-med) < 3.0*MAD:
                filtered.append(p)
        return np.array(filtered)

    top = filter_outliers(top, 1)
    bottom = filter_outliers(bottom, 1)

    left = filter_outliers(left, 0)
    right = filter_outliers(right, 0)

    # return points that consist each edge of the screen
    return ((top, bottom), (left, right))


def longest_lines(lines):
    '''
    Return longest HoughLine segments on each edge
    '''
    horz = []
    vert = []
    # split into horizontal and vertical lines
    for x1, y1, x2, y2 in lines:
        # if change in x is less than in y, then classify as vertical
        if np.abs(x1-x2) < np.abs(y1-y2):
            vert.append((x1, y1, x2, y2))
        else:
            horz.append((x1, y1, x2, y2))

    # find (very) approximate center of edges
    horz = np.array(horz)
    vert = np.array(vert)
    x_center = (horz.mean(axis=0)[0] + horz.mean(axis=0)[2]) / 2
    y_center = (vert.mean(axis=0)[1] + vert.mean(axis=0)[3]) / 2

    # split horizontal lines into top and bottom
    top = []
    bottom = []
    for x1, y1, x2, y2 in horz:
        if y1 < y_center:
            top.append((x1, y1, x2, y2))
        else:
            bottom.append((x1, y1, x2, y2))

    # split vertical lines into left and right
    left = []
    right = []
    for x1, y1, x2, y2 in vert:
        if x1 > x_center:
            right.append((x1, y1, x2, y2))
        else:
            left.append((x1, y1, x2, y2))

    # find longest line in each group
    groups = [np.array(top), np.array(bottom), np.array(left), np.array(right)]
    longest_group = []
    for group in groups:
        longest = group[np.argmax(np.linalg.norm(group[0:2]-group[2:4],
                                                 axis=1))]
        longest_group.append(longest)

    longest_group = np.array(longest_group)
    # return longest lines on each edge of the screen
    return ((np.vstack((longest_group[0, 0:2], longest_group[0, 2:4])),
             np.vstack((longest_group[1, 0:2], longest_group[1, 2:4]))),
            (np.vstack((longest_group[2, 0:2], longest_group[2, 2:4])),
             np.vstack((longest_group[3, 0:2], longest_group[3, 2:4]))))


def fit_robust_and_get_lines(img, groups):
    '''
    Fits line to each group using RANSAC algorithm, then returns
    line segments that span the image.
    '''
    horz = groups[0]
    vert = groups[1]

    model_ransac = linear_model.RANSACRegressor(linear_model
                                                .LinearRegression(),
                                                max_trials=500)
    lines = []
    for group in horz:
        X = group[:, 0].reshape(len(group[:, 0]), 1)
        Y = group[:, 1].reshape(len(group[:, 1]), 1)
        model_ransac.fit(X, Y)
        x1 = 0
        x2 = len(img[0, :]) - 1
        y1 = int(model_ransac.predict(x1)[0, 0])
        y2 = int(model_ransac.predict(x2)[0, 0])
        lines.append((x1, y1, x2, y2))
    for group in vert:
        X = group[:, 1].reshape(len(group[:, 1]), 1)
        Y = group[:, 0].reshape(len(group[:, 0]), 1)
        model_ransac.fit(X, Y)
        y1 = 0
        y2 = len(img[:, 0]) - 1
        x1 = int(model_ransac.predict(y1)[0, 0])
        x2 = int(model_ransac.predict(y2)[0, 0])
        lines.append((x1, y1, x2, y2))

    return lines

def fit_edges(groups):
    '''
    Takes grouped points and fits lines to them
    '''
    horz = groups[0]
    vert = groups[1]

    h_fits = []
    v_fits = []
    for group in horz:
        # fit line as function of x values
        h_fits.append(np.polyfit(group[:, 0], group[:, 1], 1))
    for group in vert:
        # fit line as function of y values
        v_fits.append(np.polyfit(group[:, 1], group[:, 0], 1))

    return h_fits, v_fits


def get_lines(img, coefs):
    '''
    Takes linear fits and gets endpoints within the image
    '''
    horz_coefs = coefs[0]
    vert_coefs = coefs[1]

    lines = []

    # get endpoints for horz lines
    for coef in horz_coefs:
        x1 = 0
        x2 = len(img[0, :]) - 1
        y1 = int(coef[1] + x1*coef[0])
        y2 = int(coef[1] + x2*coef[0])
        lines.append((x1, y1, x2, y2))

    # get endpoints for vert lines
    for coef in vert_coefs:
        y1 = 0
        y2 = len(img[:, 0]) - 1
        x1 = int(coef[1] + y1*coef[0])
        x2 = int(coef[1] + y2*coef[0])
        lines.append((x1, y1, x2, y2))

    return lines


def find_corners(img, lines):
    '''
    Find corners of screen within image based on edge lines.
    '''
    max_x = len(img[0, :]) - 1
    max_y = len(img[:, 1]) - 1

    corners = []

    def intersects(l1, l2):
        '''
        Return point of intersection between two lines.
        Return None if they don't intersect.
        '''
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if d != 0:
            pt_x = ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / d
            pt_y = ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / d
            if pt_x >= 0 and pt_x <= max_x and pt_y >= 0 and pt_y <= max_y:
                return (pt_x, pt_y)
            else:
                return None
        else:
            return None
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            corner = intersects(lines[i], lines[j])
            if corner:
                corners.append(corner)
    return corners


def sort_corners(corners):
    '''
    Sort corners so that they are in clockwise order starting from the top-left
    '''
    ctr_x, ctr_y = np.array(corners).mean(axis=0)
    srtd_corners = list(corners)
    for x, y in corners:
        if x < ctr_x and y < ctr_y:
            srtd_corners[0] = (x, y)
        elif x > ctr_x and y < ctr_y:
            srtd_corners[1] = (x, y)
        elif x > ctr_x and y > ctr_y:
            srtd_corners[2] = (x, y)
        elif x < ctr_x and y > ctr_y:
            srtd_corners[3] = (x, y)
    return srtd_corners


def process_frame(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, bw_img) = cv2.threshold(gray_img, 128, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(bw_img, threshold1=thresh, threshold2=thresh*1.5,
                      apertureSize=3)
    minLineLength = 100
    maxLineGap = 15
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, None,
                            minLineLength, maxLineGap)
    try:
        plt_lines = get_lines(img, fit_edges(group_lines(lines[0])))
    except:
        return None

    corners = find_corners(img, plt_lines)
    if len(corners) != 4:
        return None
    corners = sort_corners(corners)

    dest_corners = [(0, 0), (1279, 0), (1279, 719), (0, 719)]
    trans = cv2.getPerspectiveTransform(np.array(corners).astype('float32'),
                                        np.array(dest_corners)
                                        .astype('float32'))
    return cv2.warpPerspective(img, trans, (1280, 720))


def find_corner_points(points):
        corners = {(0, 0): None,
                   (1919, 0): None,
                   (1919, 1079): None,
                   (0, 1079): None}
        for corner in corners:
            # find point closest to each corner
            corners[corner] = points[np.argmin(np.linalg.norm(points-corner,
                                                              axis=1))]
        return corners


def process_frame_lines(img, lighting):
    opt = {}
    if lighting == 'dark':
        opt['green_prev'] = 0.6
    elif lighting == 'light':
        opt['green_prev'] = 0.45
    summed = img.sum(axis=2)
    summed = np.dstack((summed, summed, summed))
    warnings.filterwarnings("ignore")
    norm = np.divide(img.astype('float32'), summed)
    warnings.filterwarnings("default")
    green_img = ((norm[:, :, 1] > opt['green_prev'])).astype('uint8')
    green_img[green_img == 1] = 255
    green_img = cv2.GaussianBlur(green_img, (9, 9), 2, None, 2)
    minLineLength = 100
    maxLineGap = 15
    lines = cv2.HoughLinesP(green_img, 1, np.pi/180, 80, None,
                            minLineLength, maxLineGap)
    if lines is None:
        return None
    try:
        plt_lines = get_lines(img, fit_edges(group_lines(lines[0])))
    except:
        return None

    corners = find_corners(img, plt_lines)
    if len(corners) != 4:
        return None
    corners = sort_corners(corners)

    dest_corners = [(0, 0), (1279, 0), (1279, 719), (0, 719)]
    trans = cv2.getPerspectiveTransform(np.array(corners).astype('float32'),
                                        np.array(dest_corners)
                                        .astype('float32'))
    return cv2.warpPerspective(img, trans, (1280, 720))


def process_frame_circles(img):
    # green_img = (img[:, :, 1])
    green_img = ((img[:, :, 1] > 200) & (img[:, :, 0] < 200) &
                 (img[:, :, 2] < 200)).astype('uint8')
    green_img[green_img == 1] = 255
    green_img = cv2.GaussianBlur(green_img, (9, 9), 2, None, 2)
    _, bw_img = cv2.threshold(green_img, 120, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(bw_img, cv2.cv.CV_HOUGH_GRADIENT, 1, 200,
                               param1=200, param2=5, minRadius=10,
                               maxRadius=100)
    if circles is None:
        return None
    elif len(circles[0]) < 4:
        return None

    points = circles[0, :, 0:2]
    matches = find_corner_points(points)
    dest = np.array(matches.keys()).astype('float32')
    dest[:, 0] = dest[:, 0]*(1279./1919)
    dest[:, 1] = dest[:, 1]*(719./1079)
    src = np.array(matches.values())
    trans = cv2.getPerspectiveTransform(src, dest)
    return cv2.warpPerspective(img, trans, (1280, 720))
