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
import time


def generator_8x8(input_gen=20, leaky_alpha=0.2,dropRate=0.3,output_img=(8,8,3)):
    model = Sequential()

    model.add(Dense(32*2*2,input_dim=input_gen, name="Dens1_generator_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32 * 2 * 2, name="Dens2_generator_8x8"))
    model.add(Dropout(dropRate))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((2, 2, 32), input_shape=(32 * 2 * 2,)))

    model.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2),padding='same', name="ConvTransp1_generator_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), name="Conv1_generator_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2),padding='same', name="ConvTransp2_generator_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), name="Conv2_generator_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2D(3, kernel_size=(3,3), padding='same', activation='tanh', strides=(1,1), name="Conv3_generator_8x8"))
    
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def generator_16x16(input_gen=20, leaky_alpha=0.2,dropRate=0.3,output_img=(16,16,3)):
    model = Sequential()

    model.add(Conv2DTranspose(256, kernel_size=(3,3), strides=(2,2),padding='same', name="convTransp1_generator_16x16"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', strides=(1,1), name="conv1_generator_16x16"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', strides=(1,1), name="conv2_generator_16x16"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2D(3, kernel_size=(3,3), padding='same', activation='tanh', strides=(1,1), name="conv3_generator_16x16"))
    
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def discriminator_16x16(leaky_alpha=0.2, dropRate=0.3, image_shape=(16,16,3), output_dim=(8,8,3)):
    model = Sequential()
    
    model.add(Conv2D(512, (3, 3),padding='same',input_shape=image_shape, strides=(1,1), name="conv1_discri_16x16"))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(512, (3, 3), strides=(1,1),padding='same', name="conv2_discri_16x16"))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1,1),padding='same', name="conv3_discri_16x16"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, (3, 3), strides=(1,1),padding='same', name="conv4_discri_16x16"))
    model.add(LeakyReLU(alpha=0.2))

    #Pour coordonner les 2 models
    model.add(Conv2D(3, (3, 3), strides=(1,1),padding='same', name="conv5_discri_16x16"))
    model.add(LeakyReLU(alpha=0.2))

    model.summary()
    sgd=SGD(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def discriminator_8x8(leaky_alpha=0.2, dropRate=0.3, image_shape=(8,8,3)):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3),padding='same',input_shape=image_shape, strides=(1,1), name="conv1_discri_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, (3, 3), strides=(1,1),padding='same', name="conv2_discri_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Conv2D(128, (3, 3), strides=(1,1),padding='same', name="conv3_discri_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, (3, 3), strides=(1,1),padding='same', name="conv4_discri_8x8"))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, name="Dens1_discri_8x8"))
    model.add(Dropout(dropRate))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(1, name="Dens2_discri_8x8"))
    model.add(Activation('sigmoid'))
    model.summary()
    sgd=SGD(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def DCGAN(input_gen,dataset):
    # generator
    #g_1 = generator_model(sample_size, 0.2)
    #On load le generator_1
    #g_1.load_weights('g.h5')
    #g_2 = generator_model_part1(sample_size, 0.2)
    #On ajoute la seconde partie
    #g = Sequential([g_1,g_2])

    #g.summary()
    g_8x8    = generator_8x8(input_gen,0.2)#generatorV1_model(input_gen, 0.2)
    g_8x8.load_weights('g.h5')
    g_16x16  = generator_16x16()
    g = Sequential([g_8x8, g_16x16])
    sgd=SGD()
    g.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss='binary_crossentropy',metrics=['accuracy'])
    # discriminator
    d_8x8 = discriminator_8x8()
    d_8x8.load_weights('d.h5')
    d_16x16 = discriminator_16x16()
    #d.load_weights('d.h5')
    d_8x8.trainable = False
    d_16x16.trainable = False
    d = Sequential([d_16x16,d_8x8])
    sgd=SGD()
    d.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss='binary_crossentropy',metrics=['accuracy'])
    # GAN
    gan = Sequential([g, d])
    gan.summary()
    sgd=SGD()
    gan.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss='binary_crossentropy',metrics=['accuracy'])
    return gan, g, d


def generatorV1_model(input_gen=20, leaky_alpha=0.2,dropRate=0.3):
    model = Sequential()

    model.add(Dense(input_dim=input_gen, output_dim=2048))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32 * 8 * 8))
    model.add(Dropout(dropRate))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((8, 8, 32), input_shape=(32 * 8 * 8,)))
    #model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2),padding='same'))
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(Dropout(dropRate))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1)))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2),padding='same'))
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(Dropout(dropRate))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1)))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(UpSampling2D(size=(2, 2)))
    
    model.add(Conv2DTranspose(256, kernel_size=(4,4), strides=(2,2),padding='same'))
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(Dropout(dropRate))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=(3,3), padding='same', strides=(1,1)))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(3, kernel_size=(4,4), padding='same', activation='tanh', strides=(1,1)))
    #model.add(LeakyReLU(alpha=0.2))  

    model.summary()
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
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
    model.add(Conv2D(64, (3, 3),
               padding='same',
               input_shape=(64, 64, 3), strides=(1,1)))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))

    model.add(Conv2D(64, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Conv2D(128, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    model.add(Conv2D(128, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    model.add(Conv2D(256, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization(momentum=0.8))

    #model.add(Dropout(dropRate))
    model.add(Conv2D(512, (3, 3), strides=(1,1),padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(dropRate))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization(momentum=0.8))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(dropRate))
    model.add(LeakyReLU(alpha=0.2))
    
    #model.add(BatchNormalization(momentum=0.8))
    #model.add(Dropout(dropRate))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    sgd=SGD(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['binary_accuracy'])
    return model

def entrainement(epochs, nbrImageEntrainement, datasetImg, ChargeSauvegarde, epoch_start,pasEntrainement,input_gen,nbrColImgGen,nbrLigneColImgGen,epoch_report):
    #On crée le GAN
    (gan, g, d) = DCGAN(input_gen,datasetImg)

    if(ChargeSauvegarde == True):
        gan.load_weights('GAN.h5')
        g.load_weights('g.h5')
        d.load_weights('d.h5')

    #On doit entraîner un peu le discriminateur
    if(epoch_start == 0):
        d.trainable = True
        d.fit(datasetImg[0:nbrImageEntrainement],np.ones(nbrImageEntrainement),epochs=5)
    d.trainable = False


    #On se prépare un vecteur de bruit fixe pour pouvoir voir l'évolution de ce vecteur
    bruitFixe = np.random.rand(nbrColImgGen*nbrLigneColImgGen,input_gen)

    moyaccDiscriTrueImageArray = []
    moyAccGANArray  = []
    moyaccDiscriFalseImageArray = []

    moyLossDiscriTrueImageArray = []
    moyLossGANArray  = []
    moyLossDiscriFalseImageArray = []

    #On init la variable de stockage d'image
    imgGenereNonCalib = np.ndarray(shape=(nbrColImgGen*nbrLigneColImgGen, datasetImg.shape[1],datasetImg.shape[2],3),
                     dtype=np.float32)

    genImage = []

    for e in range(epoch_start,epochs):

        moyaccDiscriTrueImage = 0.0
        moyaccDiscriFalseImage = 0.0
        moyAccGAN = 0.0

        moyLossDiscriTrueImage = 0.0
        moyLossDiscriFalseImage = 0.0
        moyLossGAN = 0.0

        #On init le start time
        start_time = time.clock()

        for a in range(0,nbrImageEntrainement,pasEntrainement):

            temps_restant = (float(nbrImageEntrainement - a) * (time.clock() - start_time) )
            temps_restant_formate = time.strftime('%H:%M:%S', time.gmtime(temps_restant))
            print("Epochs " + str(e) + "/" + str(epochs) + " image : " + str(a) + "/" + str(nbrImageEntrainement) + 
            " temps restant = " + temps_restant_formate, end="\r")

            #On démarre le chrono
            start_time = time.clock()

            bruit = np.random.rand(pasEntrainement,input_gen)
            #On génère l'image à partir du bruit
            genImage = g.predict(bruit)

            #On entraîne le discriminateur
            d.trainable = True
            discriTrueImage = d.fit(datasetImg[a:a+pasEntrainement], np.ones(pasEntrainement),verbose=0)
            discriFalseImage = d.fit(genImage, np.zeros(pasEntrainement),verbose=0)
            d.trainable = False
            #On entraîne le generateur
            ganHistoryImg = gan.fit(bruit, np.ones(pasEntrainement),verbose=0)
            
            moyaccDiscriTrueImage   = moyaccDiscriTrueImage     + (discriTrueImage.history['acc'][0])
            moyaccDiscriFalseImage  = moyaccDiscriFalseImage    + (discriFalseImage.history['acc'][0])
            moyAccGAN               = moyAccGAN                 + (ganHistoryImg.history['acc'][0])
        
        
            #Pareil pour le loss
            moyLossDiscriTrueImage  = moyLossDiscriTrueImage    + (discriTrueImage.history['loss'][0])
            moyLossDiscriFalseImage = moyLossDiscriFalseImage   + (discriFalseImage.history['loss'][0])
            moyLossGAN              = moyLossGAN                + (ganHistoryImg.history['loss'][0])
        
        
            if ( (a % epoch_report == 0) & (a != 0) ):

                epoch_temp = (a / nbrImageEntrainement) + e

                moyaccDiscriTrueImage   = moyaccDiscriTrueImage / epoch_report
                moyAccGAN               = moyAccGAN / epoch_report
                moyaccDiscriFalseImage  = moyaccDiscriFalseImage / epoch_report
                moyLossDiscriTrueImage  = moyLossDiscriTrueImage / epoch_report
                moyLossGAN              = moyLossGAN / epoch_report
                moyLossDiscriFalseImage = moyLossDiscriFalseImage / epoch_report

                #On fait la moyenne de l'acc
                moyaccDiscriTrueImageArray.append([epoch_temp, moyaccDiscriTrueImage])
                moyAccGANArray.append([epoch_temp,moyAccGAN])
                moyaccDiscriFalseImageArray.append([epoch_temp, moyaccDiscriFalseImage])
                #Pareil pour le loss
                moyLossDiscriTrueImageArray.append([epoch_temp, moyLossDiscriTrueImage])
                moyLossGANArray.append([epoch_temp,moyLossGAN])
                moyLossDiscriFalseImageArray.append([epoch_temp, moyLossDiscriFalseImage])
                
                if( e >= 1):
                    #Sauvegarde
                    gan.save_weights('GAN.h5')
                    g.save_weights('g.h5')
                    d.save_weights('d.h5')

                #On enregistre les perfs du GAN acc
                Divers.SauvegardePerfGAN(epoch_temp,np.array(moyaccDiscriTrueImageArray),np.array(moyAccGANArray),np.array(moyaccDiscriFalseImageArray),"AccGAN")

                #On enregistre les perfs du GAN loss
                Divers.SauvegardePerfGAN(epoch_temp,np.array(moyLossDiscriTrueImageArray),np.array(moyLossGANArray),np.array(moyLossDiscriFalseImageArray),"LossGAN") 


                bruit = np.random.rand((nbrColImgGen*nbrLigneColImgGen),input_gen)
                #On génère l'image à partir du bruit
                genImage = g.predict(bruit)
                imgGenereNonCalib = copy.copy(genImage)
                imgGenerePrAffichage = Divers.UndoCalibrationValeurPixelDataset(imgGenereNonCalib)

                #On enregistre l'image générée
                Divers.SauvegardeImageMatplot(nbrColImgGen,nbrLigneColImgGen,imgGenerePrAffichage,"Resultat/ImageGenerees/epochs_" + str(epoch_temp) + ".png")
                #On enregistre les images générées avec le bruit fixe
                
                #On génère l'image à partir du bruit
                genImage = g.predict(bruitFixe)
                imgGenereNonCalib = copy.copy(genImage)
                imgGenerePrAffichage = Divers.UndoCalibrationValeurPixelDataset(imgGenereNonCalib)
                Divers.SauvegardeImageMatplot(nbrColImgGen,nbrLigneColImgGen,imgGenerePrAffichage,"Resultat/ImageGenerees/bruitFixe/epochs_" + str(epoch_temp) + ".png")

                #Si le discriminateur se trompe plus de la moitié du temps on arrete
                if(moyLossDiscriTrueImage >= 0.5):
                    print("FIN entrainement !")
                    return

def afficheMeilleurImageGAN(nombreImg,nbrColonne,nbrLigne,nomFichier,input_gen,pourcentage_reussite):

    bruit = np.random.rand(nombreImg+1,input_gen)

    #On init la variable de stockage d'image
    dataset = np.ndarray(shape=(nbrColonne*nbrLigne, 64,64,3),
                     dtype=np.float32)

    #On crée le GAN
    (gan, g, d) = DCGAN(input_gen,dataset)

    gan.load_weights('GAN.h5')
    g.load_weights('g.h5')
    d.load_weights('d.h5')

    imgListe = g.predict(bruit)
    pourcentageReussite = gan.predict(bruit)

    i=0
    index = 0
    while(i < ((nbrColonne*nbrLigne)-2) | index < nombreImg):
        print("Images trouvés : " + str(i) + " / " + str(nbrColonne*nbrLigne) + "   Image parcourue : " + str(index) + " / " + str(nombreImg), end="\r")
        
        if pourcentageReussite[index] > pourcentage_reussite:
            if i < (nbrColonne*nbrLigne) :
                dataset[i] = imgListe[index]
            i+=1
        index+=1
    dataset = Divers.UndoCalibrationValeurPixelDataset(dataset)

    Divers.SauvegardeImageMatplot(nbrColonne,nbrLigne,dataset,"Resultat/"+nomFichier)



