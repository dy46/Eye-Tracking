import Tkinter, Tkconstants, tkFileDialog
from Tkinter import *
from PIL import ImageTk, Image
import os
import cv2
import crop

button_opt = {'fill': Tkconstants.BOTH, 'padx': 10, 'pady': 10, 'side': 'left'}
selected = 0
cnt = 0
panel=[]
refimg=[]
flag=False
ready1=False
ready2=False
s=[]
multiFlag=None
imgname=[]

def asksaveasfilename(self):

    """Returns an opened file in write mode.
    This time the dialog just returns a filename and the file is opened by your own code.
    """

    # get filename
    filename = tkFileDialog.asksaveasfilename(**self.file_opt)

    # open file on your own
    if filename:
      return open(filename, 'w')

def askdirectory(self):

    """Returns a selected directoryname."""

    return tkFileDialog.askdirectory(**self.dir_opt)

def messagealert(message):
    Alert = Tk()
    Alert.title("Alert")
    AlertWindow = Message(Alert, text=message, padx=5, pady=5, relief=RAISED, width=4000)
    AlertWindow.pack()
    Alert.mainloop()

def askopenimage():
    global bottomframe, cnt, selected, refimg, panel, ready1

    if selected>=cnt:
        messagealert("Hey! You need to add an object first!")
        return
    img_opt = options = {}
    options['defaultextension'] = '.png'
    options['filetypes'] = [('all files', '.*'), ('text files', '.png')]
    options['initialdir'] = './'
    options['initialfile'] = 'ipuprofen.png'
    options['parent'] = bottomframe
    options['title'] = 'File Chooser'


    filename = tkFileDialog.askopenfilename(**img_opt)
    refimg.append(filename)
    img2 = ImageTk.PhotoImage(resized(Image.open(filename)))

    panel[selected*2].configure(image = img2)
    panel[selected*2].image = img2

    selected+=1
    if selected>=cnt:
        ready1=True

    updatestatus()

def resized(originalImg):
    return originalImg.resize((250, 125),Image.ANTIALIAS)



def askopenvideo():
    global var, bottomframe, cnt, ready2, filename, path
    video_opt = options2 = {}
    options2['defaultextension'] = '.mp4'
    options2['filetypes'] = [('all files', '.*'), ('text files', '.mp4')]
    options2['initialdir'] = './'
    options2['initialfile'] = 'ipuprofen.mp4'
    options2['parent'] = bottomframe
    options2['title'] = 'File Chooser'

    ready2=True
    filename = tkFileDialog.askopenfilename(**video_opt)
    path = os.path.abspath(os.path.expanduser(filename))
    var.set(path)

def runandquit():
    global flag, root
    if multiFlag==None:
        messagealert('Hey! You have not selected how to process the video')
    if not ready1:
        messagealert("Hey! You have not selected enough objects!"+str(selected)+str(cnt))
        return
    if not ready2:
        messagealert("Hey! You have not selected the video!")
        return
    root.quit()

def grayout():
    global panel, imgname
    #print 'ImHere'
    temp=StringVar()
    for i in range(0,len(panel)):
        if i % 2==1:
            #print i
            print imgname[i/2].get()
            #panel[i]=Tkinter.Entry(bottomframe,textvariable=imgname[0])
            panel[i].configure(state=DISABLED)


def addobject():
    global bottomframe, cnt, frame, panel, imgname
    imgname.append(StringVar());
    if cnt>=4:
        messagealert("Hey! That's too many objects")
    cnt+=1
    initimg1=ImageTk.PhotoImage(resized(Image.open("logo.png")))
    #tr=Tkinter.Entry(bottomframe, bd=5)
    #tr.pack(side=LEFT)
    panel.append(Tkinter.Label(bottomframe,image=initimg1))
    panel.append(Tkinter.Entry(bottomframe,textvariable=imgname[-1]))
    panel[-2].img=initimg1
    if cnt==1:
        conf="left"
    if cnt==2:
        conf="right"
    if cnt==3:
        conf="bottom"
    if cnt==4:
        conf="right"
    panel[-1].grid(row=(cnt-1)/2*2,column=(cnt-1)%2)
    panel[-2].grid(row=(cnt-1)/2*2+1,column=(cnt-1)%2)
    ready1=False
    updatestatus();

def updatestatus():
    global cnt, selected, status
    status.set(str(selected)+"/"+str(cnt))

def sel():
    global multiFlag
    selection = "Selected " + choice.get()
    if processingdict[choice.get()]==1: 
        multiFlag=False
    else:
        multiFlag=True
    #rbutton.config(text = selection)

def addcrop():
    global bottomframe, cnt, selected, refimg, panel, ready1, filename, path

    #if selected>=cnt:
    #    messagealert("Hey! You need to add an object first!")
    #    return
    if not ready2:
        messagealert("Hey! You need to select the video first")
    filename1=filename.split("/")[-1]
    print filename1

    v=cv2.VideoCapture(filename1)
    ret,imgForCrop=v.read()
    imgname=crop.crop(imgForCrop)
    for f in imgname:
        if (selected>=cnt):
            addobject()
        print f
        refimg.append(f)
        img2 = ImageTk.PhotoImage(resized(Image.open(f)))
        print selected
        panel[selected].configure(image = img2)
        panel[selected].image = img2

        selected+=1
        if selected>=cnt:
            ready1=True

        updatestatus()

def start():
    global imgname, frame, bottomframe, panel, panel2, var, cnt, status, flag, root, choice, processingdict, rbutton, multiFlag

    root = Tkinter.Tk()
    root.title("Eye-Tracking and Food Choices")

    #Add a background image
    background_image= ImageTk.PhotoImage(Image.open("bg.jpg"))
    background_label = Tkinter.Label(root, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    """
    w = 400 # width for the Tk root
    h = 600 # height for the Tk root

    # get screen width and height
    ws = root.winfo_screenwidth() # width of the screen
    hs = root.winfo_screenheight() # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)

    # set the dimensions of the screen 
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    """

    frame = Tkinter.Frame(root)
    frame.pack(side=TOP)
    
    middle = Tkinter.Frame(root)
    middle.pack(side=TOP)

    bottomframe = Frame(root)
    bottomframe.pack(side=BOTTOM)

    Tkinter.Button(frame, text='Select Object', command=askopenimage).pack(**button_opt)
    Tkinter.Button(frame, text='Select Video', command=askopenvideo).pack(**button_opt)
    Tkinter.Button(frame, text='  Run  ', command=runandquit).pack(**button_opt)
    Tkinter.Button(frame, text='Add Object', command=addobject).pack(**button_opt)
    Tkinter.Button(frame, text='Crop Image', command=addcrop).pack(**button_opt)
    
    processingdict={"Single Processing":1,"Multi Processing":2}
    #Add a status for single or multi processing
    choice=StringVar()
    R1=Radiobutton(frame,text="Single Processing",variable=choice, value="Single Processing", command=sel)
    R1.pack(anchor=W)
    R2=Radiobutton(frame,text="Multi Processing",variable=choice, value="Multi Processing", command=sel)
    R2.pack(anchor=W)
    rbutton=Label(frame)
    rbutton.pack(side=RIGHT)

    #Add a status bar for image
    status = StringVar()
    statuspanel = Tkinter.Label(middle, textvariable = status, relief=RAISED)
    status.set("0/1")
    statuspanel.pack(side=LEFT)


    #Add a status bar for video 
    var = StringVar()
    panel2 = Tkinter.Label(middle, textvariable = var, relief=RAISED)
    var.set("No Video Now")
    panel2.pack(side=LEFT)

    #Add a Reference Image.
    cnt=0 
    addobject()

    objectname = Tkinter.Button(middle, text='Finish', command=grayout)
    objectname.pack(side=LEFT)
    root.mainloop()
    result=[]
    for i in range(0,len(imgname)):
        result.append(imgname[i].get())
    root.destroy()
    return refimg, filename, multiFlag, result

if __name__=="__main__":
    list1=start()
    print list1
