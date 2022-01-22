from pygame import image
import pygame,sys
from pygame.locals import *
import numpy as np
import matplotlib.pyplot as plt 
from keras.models import load_model
import cv2

WINDOW_SIZE_X = 640
WINDOW_SIZE_Y = 480
BOUNDARY = 5

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)

MODEL = load_model('d:\project1\mymodel.h5')

LABELS = {0:'zero',
        1:'one',
        2:'two',
        3:'three',
        4:'four',
        5:'five',
        6:'six',
        7:'seven',
        8:'eight',
        9:'nine'}


pygame.init()

FONT = pygame.font.Font(None,18)

PREDICT = True

DISPLAYSURF = pygame.display.set_mode((WINDOW_SIZE_X,WINDOW_SIZE_Y))
pygame.display.set_caption('ĐOÁN CHỮ SỐ')


iswriting = False
number_xcord = []
number_ycord = []



while (True):
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

        if event.type == pygame.MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF,WHITE,(xcord,ycord),4,0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            
            rect_min_x = max(number_xcord[0]-BOUNDARY,0)
            rect_max_x = min(WINDOW_SIZE_X,number_xcord[-1]+BOUNDARY)
            
            rect_min_y = max(number_ycord[0]-BOUNDARY,0)
            rect_max_y = min(WINDOW_SIZE_Y,number_ycord[-1]+BOUNDARY)

            number_xcord = []
            number_ycord = []
            
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x,rect_min_y:rect_max_y].T.astype(np.float32)
          

            if PREDICT:
            
                image = cv2.resize(img_arr,(28,28))
                image = np.pad(image,(10,10),'constant',constant_values = 0)
                image = cv2.resize(image,(28,28))/255
               
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
                # plt.imshow(image,cmap = 'binary')
                # plt.title(label)
                
                
                textSurface = FONT.render(label,True,RED,WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left = rect_min_x
                textRecObj.top = rect_min_y

                DISPLAYSURF.blit(textSurface,textRecObj)
                pygame.draw.rect (DISPLAYSURF, RED, (rect_min_x,rect_min_y,rect_max_x-rect_min_x,rect_max_y-rect_min_y), 1) 
                # plt.show()

            

        pygame.display.update()