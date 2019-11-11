import sys
sys.path.insert(0, '../')
from tkinter import *
from PIL import Image

import numpy as np
import torch

from capsule.net import CapsNet

__all__ = [
    'Paint'
]


class Paint:

    SIZE = (300, 300) # Width, Height
    TARGET_SIZE = (28, 28) # Width, Height

    DEFAULT_LINE_WIDTH = 25
    DEFAULT_COLOR = 'black'

    PATH_TO_MODEL = '../models/capsnet_state.pth'
    DEVICE = 'cpu'

    def __init__(self):
        self.root = Tk()

        self.clear_btn = Button(self.root, text='clear', command=self.clear)
        self.clear_btn.grid(row=0, column=0)

        self.c = Canvas(self.root, bg='white', width=Paint.SIZE[0], height=Paint.SIZE[1])
        self.c.grid(row=1, columnspan=5)

        self.setup()

    def run(self):
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = Paint.DEFAULT_LINE_WIDTH
        self.color = Paint.DEFAULT_COLOR

        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

        self.net = CapsNet()
        self.net.load_state_dict(torch.load(open(Paint.PATH_TO_MODEL, 'rb'), map_location=torch.device(Paint.DEVICE)))

    def paint(self, event):        
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.color,
                               capstyle=ROUND, smooth=True, splinesteps=36)
                
        self.old_x = event.x
        self.old_y = event.y

    def clear(self):
        self.c.delete('all')

    def save_load_image(self):
        # Save PS
        self.c.postscript(file='out.eps')

        # Load NP Array
        self.image = Image.open('out.eps').convert('RGB').convert('LA').resize(Paint.TARGET_SIZE).getdata(band=0)
        self.image = np.reshape(self.image, Paint.TARGET_SIZE)

    def run_capsnet(self):
        self.save_load_image()
        input = torch.tensor(self.image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        _, norm, _ = self.net(input)
        print(norm.argmax().item())

    def reset(self, event):
        # CapsNet
        self.run_capsnet()

        # Reset
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint().run()
