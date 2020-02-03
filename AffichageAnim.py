from tkinter import *
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageTk 

#fenetre = Tk()
TAILLE = [200,200]
#curseurArray = np.zeros(11)#np.ndarray(shape=(11,1))
Valeur = np.ndarray(shape=(2,10))

#Changement de valeur d'un scale
def afficherImage(fenetre,image_,cadre):
    #im=Image.open("1.png")
    im = Image.fromarray(image_,'RGB')
    photo = ImageTk.PhotoImage(im,size=TAILLE[0]*TAILLE[1])
    item = cadre.create_image(32,32,image =photo)

def checkValeur():
    for i in range(10):
        Valeur[0][i] = curseurArray[i].get()
        Valeur[1][i] = 0



def init(fenetre):
    curseurArray = [Scale(fenetre, orient='horizontal', from_=0, to=1,
    resolution=0.01, tickinterval=2, length=500,label='Param 0')]

    for i in range(10):
        curseurArray.append(Scale(fenetre, orient='horizontal', from_=0, to=1,
        resolution=0.01, tickinterval=2, length=500,label='Param '+str(i)))#,command=maj,variable=curseurArray[i])#,variable=curseurArray[i],command=maj)
        curseurArray[i+1].pack()


    cadre=Canvas(fenetre,width=TAILLE[0],height=TAILLE[1],bg="white")
    cadre.pack()
    return cadre,curseurArray



from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, \
    Flatten, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D,MaxPooling2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam, RMSprop,SGD
from keras.initializers import RandomNormal
from keras.models import load_model
import matplotlib.pyplot as plt
import time

def generator_model(nbrParamEntree=10, dropRate=0.3,latent_dim=100, leaky_alpha=0.2):
    model = Sequential()
    
    model.add(Dense(32, input_shape=(nbrParamEntree,)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=leaky_alpha))

    model.add(Dense(64*64*3))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=leaky_alpha))


    # (None,16*16*128)>>(None,16,16,128)
    model.add(Reshape((64, 64, 3)))

    
    # (None,16,16,128)>>(None,32,32,256)
    model.add(Conv2D(128, kernel_size=(2,2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropRate))

    model.add(Conv2D(128, kernel_size=(2,2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropRate))

    #(None,32,32,256)>>(None,32,32,256)
    model.add(Conv2D(3, kernel_size=(2,2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("sigmoid"))   
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

import random
import sys
from threading import Thread
import time

def convImg(img):
    img = abs(img) * 255.
    img = img.astype('uint8')
    img = np.asarray(img)

    return img


class Afficheur(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self,fenetre,cadre,generator):
        Thread.__init__(self)
        self.fenetre = fenetre
        self.cadre = cadre
        self.generator = generator

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        while True:
            print("Check valeur")
            checkValeur()
            print(Valeur)

            noise = np.random.normal(0, 1, (2, 10,))
            #print(noise.shape)
            
            img = generator.predict(noise)
##            img = img * 0.5 + 0.5
##            img = img * 255
##            img = img.astype('uint8')
        ##    plt.imshow(img)
        ##    plt.show()
            img = 0
            afficherImage(fenetre,img,cadre)
        

            time.sleep(1)
            




#while True:

fenetre = Tk()
cadre,curseurArray = init(fenetre)

### Création des threads
#thread_1 = Afficheur(fenetre,cadre,generator)
##
### Lancement des threads
#thread_1.start()

noise = np.random.normal(0, 1, (2, 10))

#img = generator.predict(noise)
##plt.imshow(noise)
##plt.show()
##checkValeur()
###img = generator.predict(Valeur)[0]
##img = 0
#afficherImage(fenetre,noise,cadre)

##im = Image.fromarray(noise,'RGB')
##photo = ImageTk.PhotoImage(im,size=64*64)
##item = cadre.create_image(32,32,image =photo)

generator = generator_model(10,0.2)
generator.load_weights('g.h5')
#generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])

from resizeimage import resizeimage

while True:
    #noise = np.random.normal(0, 1, (64, 64,3))
    checkValeur()
    j = generator.predict(Valeur)[0]
    im = convImg(j)
    im = Image.fromarray(im,'RGB')
    #im = resizeimage.resize_thumbnail(im, TAILLE)
    im = im.resize(TAILLE)
    photo = ImageTk.PhotoImage(im,size=TAILLE[0]*TAILLE[1])
    #photo = photo.zoom(2) # zoom x2
    item = cadre.create_image(int(TAILLE[0]/2),int(TAILLE[1]/2),image =photo)
    #afficherImage(fenetre,im,cadre)
    fenetre.update()
    #time.sleep(2)
#fenetre.mainloop()


    

