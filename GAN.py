from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, \
    Flatten, LeakyReLU, BatchNormalization, Conv2DTranspose, Conv2D,MaxPooling2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import Adam, RMSprop,SGD
from keras.initializers import RandomNormal
from keras.models import load_model
import numpy as np
import Divers
import matplotlib.pyplot as plt
import copy


def DCGAN(input_gen,dataset):
    # generator
    #g_1 = generator_model(sample_size, 0.2)
    #On load le generator_1
    #g_1.load_weights('g.h5')
    #g_2 = generator_model_part1(sample_size, 0.2)
    #On ajoute la seconde partie
    #g = Sequential([g_1,g_2])

    #g.summary()
    g = generatorV1_model(input_gen, 0.2)
    # discriminator
    d = discriminator_model(image_shape=(dataset.shape[1],dataset.shape[2],dataset.shape[3]))
    #d.load_weights('d.h5')
    d.trainable = False
    # GAN
    gan = Sequential([g, d])
    
    sgd=SGD()
    gan.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss='binary_crossentropy',metrics=['accuracy'])
    return gan, g, d


def generatorV1_model(input_gen=20, leaky_alpha=0.2,dropRate=0.3):
    model = Sequential()

    model.add(Dense(input_dim=input_gen, output_dim=2048))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32 * 8 * 8))
    #model.add(Dropout(dropRate))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 32), input_shape=(32 * 8 * 8,)))
    #model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2),padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=(4,4), padding='same', strides=(1,1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    #model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(64, kernel_size=(4,4), strides=(2,2),padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=(2,2), padding='same', strides=(1,1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    #model.add(UpSampling2D(size=(2, 2)))
    
    model.add(Conv2DTranspose(32, kernel_size=(2,2), strides=(2,2),padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(3, kernel_size=(4,4), padding='same', activation='tanh', strides=(1,1)))
    #model.add(LeakyReLU(alpha=0.2))  

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

#def discriminatorV1_model(leaky_alpha=0.2, dropRate=0.3, image_shape=(64,64,3)):
#     model = Sequential()
    
#    # layer1 (None,64,64,3)>>(None,32,32,32)
#    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
#    model.add(LeakyReLU(alpha=leaky_alpha))

def generator_model_part1(dropRate=0.3, leaky_alpha=0.2):
    model = Sequential()


    # (None,16,16,128)>>(None,32,32,256)
    model.add(Conv2D(64, kernel_size=(2,2), padding="same",input_shape=(64,64,3)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropRate))

    model.add(Conv2D(64, kernel_size=(2,2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropRate))

    #(None,32,32,256)>>(None,32,32,256)
    model.add(Conv2D(3, kernel_size=(2,2), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("sigmoid"))    
    
    model.summary()
    
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model


def generator_model(nbrParamEntree=10, dropRate=0.3, leaky_alpha=0.2):
    model = Sequential()
    
    model.add(Dense(64*64*3, input_shape=(nbrParamEntree,)))
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
    
    model.summary()
    
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def discriminator_model(leaky_alpha=0.2, dropRate=0.3, image_shape=(32,32,3)):
    model = Sequential()
    
    # layer1 (None,64,64,3)>>(None,32,32,32)
    model.add(Conv2D(256, (4, 4),
               padding='same',
               input_shape=(64, 64, 3), strides=(1,1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(256, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2D(256, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization(momentum=0.8))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(Dropout(dropRate))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    sgd=SGD(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def entrainement(epochs, nbrImageEntrainement, datasetImg, ChargeSauvegarde, epoch_start,pasEntrainement,input_gen,nbrColImgGen,nbrLigneColImgGen):
    #On crée le GAN
    (gan, g, d) = DCGAN(input_gen,datasetImg)

    #On doit entraîner un peu le discriminateur
    d.trainable = True
    d.fit(datasetImg[0:nbrImageEntrainement],np.ones(nbrImageEntrainement),epochs=1)
    d.trainable = False

    if(ChargeSauvegarde == True):
        gan.load_weights('GAN.h5')
        g.load_weights('g.h5')
        d.load_weights('d.h5')

    #On se prépare un vecteur de bruit fixe pour pouvoir voir l'évolution de ce vecteur
    bruitFixe = np.random.rand(nbrColImgGen*nbrLigneColImgGen,input_gen)

    moyaccDiscriTrueImageArray = []
    moyAccGANArray  = []
    moyaccDiscriFalseImageArray = []

    moyLossDiscriTrueImageArray = []
    moyLossGANArray  = []
    moyLossDiscriFalseImageArray = []

    #On init la variable de stockage d'image
    imgGenereNonCalib = np.ndarray(shape=(50, datasetImg.shape[1],datasetImg.shape[2],3),
                     dtype=np.float32)

    genImage = []

    for e in range(epochs):

        #On clear les images saves
        #genImage.clear()

        moyaccDiscriTrueImage = 0.0
        moyaccDiscriFalseImage = 0.0
        moyAccGAN = 0.0

        moyLossDiscriTrueImage = 0.0
        moyLossDiscriFalseImage = 0.0
        moyLossGAN = 0.0


        for a in range(0,nbrImageEntrainement,pasEntrainement):

            print("Epochs " + str(e+1) + "/" + str(epochs) + " image : " + str(a+1) + "/" + str(nbrImageEntrainement), end="\r")

            bruit = np.random.rand(pasEntrainement+1,input_gen)
            #On génère l'image à partir du bruit
            genImage = g.predict(bruit)

            if(a < 50):
                imgGenereNonCalib = copy.copy(genImage)

            #On enregistre les 2 dernière images générées
            #if(a >= nbrImageEntrainement - 2):
            #    genImage.append(genImage)

            #On entraîne le discriminateur
            d.trainable = True
            discriTrueImage = d.fit(datasetImg[a:a+pasEntrainement+1], np.ones(pasEntrainement+1),validation_split=0.2,verbose=0)
            discriFalseImage = d.fit(genImage, np.zeros(pasEntrainement+1),validation_split=0.2,verbose=0)
            d.trainable = False
            #On entraîne le generateur
            ganHistoryImg = gan.fit(bruit, np.ones(pasEntrainement+1),validation_split=0.2,verbose=0)
            
            if(discriTrueImage.history['val_acc'][0] == 0):
                moyaccDiscriTrueImage = moyaccDiscriTrueImage + 0
            else:
                moyaccDiscriTrueImage = moyaccDiscriTrueImage + (discriTrueImage.history['val_acc'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))
            if(discriFalseImage.history['val_acc'][0] == 0):
                moyaccDiscriFalseImage = moyaccDiscriFalseImage + 0
            else:
                moyaccDiscriFalseImage = moyaccDiscriFalseImage + (discriFalseImage.history['val_acc'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))
            if(ganHistoryImg.history['val_acc'][0] == 0):
                moyAccGAN = moyAccGAN + 0
            else:
                moyAccGAN = moyAccGAN + (ganHistoryImg.history['val_acc'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))
        
        
            #Pareil pour le loss

            if(discriTrueImage.history['val_loss'][0] == 0):
                moyLossDiscriTrueImage = moyLossDiscriTrueImage + 0
            else:
                moyLossDiscriTrueImage = moyLossDiscriTrueImage + (discriTrueImage.history['val_loss'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))
            if(discriFalseImage.history['val_loss'][0] == 0):
                moyLossDiscriFalseImage = moyLossDiscriFalseImage + 0
            else:
                moyaccDiscriFalseImage = moyaccDiscriFalseImage + (discriFalseImage.history['val_loss'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))
            if(ganHistoryImg.history['val_loss'][0] == 0):
                moyLossGAN = moyLossGAN + 0
            else:
                moyLossGAN = moyLossGAN + (ganHistoryImg.history['val_loss'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))
        
        
        #On fait la moyenne de l'acc
        moyaccDiscriTrueImageArray.append([e, moyaccDiscriTrueImage])
        moyAccGANArray.append([e,moyAccGAN])
        moyaccDiscriFalseImageArray.append([e, moyaccDiscriFalseImage])
        #Pareil pour le loss
        moyLossDiscriTrueImageArray.append([e, moyLossDiscriTrueImage])
        moyLossGANArray.append([e,moyLossGAN])
        moyLossDiscriFalseImageArray.append([e, moyLossDiscriFalseImage])


        #Sauvegarde
        gan.save_weights('GAN.h5')
        g.save_weights('g.h5')
        d.save_weights('d.h5')

        #On enregistre les perfs du GAN acc
        Divers.SauvegardePerfGAN(e,np.array(moyaccDiscriTrueImageArray),np.array(moyAccGANArray),np.array(moyaccDiscriFalseImageArray),"AccGAN")

        #On enregistre les perfs du GAN loss
        Divers.SauvegardePerfGAN(e,np.array(moyLossDiscriTrueImageArray),np.array(moyLossGANArray),np.array(moyLossDiscriFalseImageArray),"LossGAN")
       
        #On reset les vars
        moyaccDiscriTrueImage = 0.0
        moyaccDiscriFalseImage = 0.0
        moyAccGAN = 0.0      
        moyLossDiscriFalseImage = 0.0
        moyLossDiscriTrueImage = 0.0
        moyLossGAN = 0.0 

        imgGenereNonCalib = copy.copy(genImage)
        #imgGenereNonCalib = ( (imgGenereNonCalib + 1) * 127.5)
        #imgGenereNonCalib = imgGenereNonCalib.astype(np.uint8)
        imgGenerePrAffichage = Divers.UndoCalibrationValeurPixelDataset(imgGenereNonCalib)

        #On enregistre l'image générée
        Divers.SauvegardeImageMatplot(nbrColImgGen,nbrLigneColImgGen,imgGenerePrAffichage,"Resultat/ImageGenerees/epochs_" + str(e) + ".png")
        #On enregistre les images générées avec le bruit fixe
        
        #On génère l'image à partir du bruit
        genImage = g.predict(bruitFixe)
        imgGenereNonCalib = copy.copy(genImage)
        imgGenerePrAffichage = Divers.UndoCalibrationValeurPixelDataset(imgGenereNonCalib)
        Divers.SauvegardeImageMatplot(nbrColImgGen,nbrLigneColImgGen,imgGenerePrAffichage,"Resultat/ImageGenerees/bruitFixe/epochs_" + str(e) + ".png")

        #imgGenereNonCalib = imgGenereNonCalib.astype(np.float32)




def afficheMeilleurImageGAN(nombreImg,nbrColonne,nbrLigne,nomFichier):

    bruit = np.random.rand(nombreImg+1,20)

    #On init la variable de stockage d'image
    dataset = np.ndarray(shape=(nbrColonne*nbrLigne, 64,64,3),
                     dtype=np.float32)

    #On crée le GAN
    (gan, g, d) = DCGAN(20,dataset)

    gan.load_weights('GAN.h5')
    g.load_weights('g.h5')
    d.load_weights('d.h5')

    imgListe = g.predict(bruit)
    pourcentageReussite = gan.predict(bruit)

    i=0
    index = 0
    while(i < ((nbrColonne*nbrLigne)-2) | index < nombreImg):
        print("Images trouvés : " + str(i) + " / " + str(nbrColonne*nbrLigne) + "   Image parcourue : " + str(index) + " / " + str(nombreImg), end="\r")
        
        if pourcentageReussite[index] > 0.01:
            if i < (nbrColonne*nbrLigne) :
                dataset[i] = imgListe[index]
            i+=1
        index+=1
    dataset = dataset * 255.
    dataset = dataset.astype(np.uint8)

    Divers.SauvegardeImageMatplot(nbrColonne,nbrLigne,dataset,"Resultat/"+nomFichier)



