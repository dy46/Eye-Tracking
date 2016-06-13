from PIL import Image

foreground = Image.open("2.png")
background = Image.open("frame.png")

background.paste(foreground, (50, 0), foreground)
background.show()