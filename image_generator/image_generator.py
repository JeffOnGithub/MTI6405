# -*- coding: utf-8 -*-
"""
@title: MTI-6405, virtual segmented image generator
@author: Jeff Grenier
@email: jeff@grenier.cc
"""
from random import randint
from PIL import Image
import glob, os

#CONFIGURATION
BG_FOLDER = "backgrounds/"
FG_FOLDER = "foregrounds/"
GEN_FOLDER = "virtual_images/"
MASK_FOLDER = "virtual_masks/"
IMG_EXT = ".png"
MIN_FG_IMAGES = 1
MAX_FG_IMAGES = 10
HOW_MANY_IMAGES = 10

#Image loader

#Load all backgrounds
backgrounds = []
for infile in glob.glob(BG_FOLDER + "/*" + IMG_EXT):
    file, ext = os.path.splitext(infile)
    image = Image.open(file + ext)
    backgrounds.append(image)

#Load all foregrounds
foregrounds = []
for infile in glob.glob(FG_FOLDER + "/*" + IMG_EXT):
    file, ext = os.path.splitext(infile)
    image = Image.open(file + ext)
    foregrounds.append(image)

#Image/mask generation

for a in range(0, HOW_MANY_IMAGES):
    #Load a background image
    im_bg = backgrounds[randint(0, len(backgrounds) - 1)]
    bg_width, bg_height = im_bg.size
    
    #Create a blank mask based on background size
    im_mask = Image.new('RGB', im_bg.size)
    
    #Add a random number of foreground elements
    for b in range(MIN_FG_IMAGES, randint(MIN_FG_IMAGES, MAX_FG_IMAGES) + 1):
        
        #Load a foreground image
        im_fg = foregrounds[randint(0, len(foregrounds) - 1)]
        fg_width, fg_height = im_fg.size
        
        #Convert black to transparent
        im_fg = im_fg.convert("RGBA")
        pixdata_fg = im_fg.load()
        for y in range(fg_height):
            for x in range(fg_width):
                if pixdata_fg[x, y] == (0, 0, 0, 255):
                    pixdata_fg[x, y] = (0, 0, 0, 0)
        
        #Paste foreground on background and mask using foreground as mask
        #Get a random position
        position_x = randint(0, max(0, bg_width - fg_width))
        position_y = randint(0, max(0, bg_height - fg_height))
        
        im_bg.paste(im_fg, (position_x, position_y), im_fg)
        im_mask.paste(im_fg, (position_x, position_y), im_fg)
        
        #Switch mask to black and white only
        pixdata_mask = im_mask.load()
        for y in range(bg_height):
            for x in range(bg_width):
                if pixdata_mask[x, y] != (0, 0, 0):
                    pixdata_mask[x, y] = (255, 255, 255)
    
    #Display / save the generated photo and its mask
    #im_bg.show()
    im_bg.save(GEN_FOLDER + "virtual_image-" + str(a) + ".jpg", "JPEG")
    im_mask.save(MASK_FOLDER + "virtual_mask-" + str(a) + ".png", "PNG")
