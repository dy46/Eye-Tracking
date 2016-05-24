'''
Display test for PyschoPy.  We will take a video of the screen as it
displays this test, then we will attempt to extract the screen from
the video.
'''

from psychopy import visual, core, event
import glob


def drawBorder(win, borderType):
    if borderType == 'line':
        loc = 0.98
        visual.Line(win, start=(-loc, -loc), end=(loc, -loc), autoDraw=True,
                    lineColor=[0, 255, 0], lineColorSpace='rgb255',
                    units='norm', lineWidth=60)
        visual.Line(win, start=(loc, -loc), end=(loc, loc), autoDraw=True,
                    lineColor=[0, 255, 0], lineColorSpace='rgb255',
                    units='norm', lineWidth=60)
        visual.Line(win, start=(loc, loc), end=(-loc, loc), autoDraw=True,
                    lineColor=[0, 255, 0], lineColorSpace='rgb255',
                    units='norm', lineWidth=60)
        visual.Line(win, start=(-loc, loc), end=(-loc, -loc), autoDraw=True,
                    lineColor=[0, 255, 0], lineColorSpace='rgb255',
                    units='norm', lineWidth=60)
    elif borderType == 'dot':
        visual.Circle(win, size=(0.1, 0.1*(8.0/5)),
                      fillColor=[0, 255, 0], fillColorSpace='rgb255',
                      units='norm', pos=(-0.95, -0.92), autoDraw=True,
                      lineColor=([-1, -1, -1]))
        visual.Circle(win, size=(0.1, 0.1*(8.0/5)),
                      fillColor=[0, 255, 0], fillColorSpace='rgb255',
                      units='norm', pos=(0.95, -0.92), autoDraw=True,
                      lineColor=([-1, -1, -1]))
        visual.Circle(win, size=(0.1, 0.1*(8.0/5)),
                      fillColor=[0, 255, 0], fillColorSpace='rgb255',
                      units='norm', pos=(0.95, 0.92), autoDraw=True,
                      lineColor=([-1, -1, -1]))
        visual.Circle(win, size=(0.1, 0.1*(8.0/5)),
                      fillColor=[0, 255, 0], fillColorSpace='rgb255',
                      units='norm', pos=(-0.95, 0.92), autoDraw=True,
                      lineColor=([-1, -1, -1]))


def images(borderType):
    testWin = visual.Window(
            size=(2560, 1600), monitor="tobiiMonitor", units="pix", screen=0,
            fullscr=True, color=(-1, -1, -1))

    drawBorder(testWin, borderType)

    ims = glob.glob('images/display/*')
    testWin.flip()
    core.wait(2)
    for im in ims:
        visual.ImageStim(testWin, image=im, units='norm',
                         size=(1.0, 1.0)).draw()
        testWin.flip()
        core.wait(2)

    testWin.close()


def video():
    testWin = visual.Window(
            size=(1280, 800), monitor="tobiiMonitor", units="pix", screen=0,
            fullscr=True, color=(-1, -1, -1), waitBlanking=False)
    drawBorder(testWin, 'line')
    mov = visual.MovieStim(testWin,
                           filename='images/display/Cats_Being_Jerks.mp4',
                           units='norm', size=(1.0, 1.0))

    while mov.status != visual.FINISHED:
        mov.draw()
        testWin.flip()
        if event.getKeys(keyList=['escape','q']):
            testWin.close()
            core.quit()
    testWin.close()


def circles():
    testWin = visual.Window(
            size=(1280, 800), monitor="tobiiMonitor", units="pix", screen=0,
            fullscr=True)
    x = -0.5
    y = -0.5
    dotStim = visual.Circle(testWin, size=(0.1, 0.1*(8.0/5)), fillColor=[0, 1, 0],
                            units='norm', pos=(x, y), autoDraw=True)
    testWin.flip()
    core.wait(4)
    for i in range(100):
        x = x+0.01
        dotStim.pos = (x, y)
        testWin.flip()
    for i in range(100):
        y = y+0.01
        dotStim.pos = (x, y)
        testWin.flip()
    for i in range(100):
        x = x-0.01
        dotStim.pos = (x, y)
        testWin.flip()
    for i in range(100):
        y = y-0.01
        dotStim.pos = (x, y)
        testWin.flip()
    testWin.close()


def main():
    video()

if __name__ == '__main__':
    main()
