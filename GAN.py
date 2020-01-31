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


def DCGAN(sample_size,dataset):
    # generator
    g = generator_model(sample_size, 0.2)
    #g = generatorV1_model(sample_size, 0.2)
    # discriminator
    d = discriminator_model(image_shape=(dataset.shape[1],dataset.shape[2],dataset.shape[3]))
    d.trainable = False
    # GAN
    gan = Sequential([g, d])
    
    sgd=SGD()
    gan.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss='binary_crossentropy',metrics=['accuracy'])
    return gan, g, d


def generatorV1_model(nbrParamEntree=10, leaky_alpha=0.2):
    model = Sequential()

    # layer1 (None,500)>>(None,128*16*16)
    model.add(Dense(128 * 64 * 64, activation="sigmoid", input_shape=(nbrParamEntree,)))

    # (None,16*16*128)>>(None,16,16,128)
    model.add(Reshape((64, 64, 128)))

    model.add(Conv2D(16, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(16, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(Conv2D(3, kernel_size=10, padding="same"))
    model.add(Activation("sigmoid"))
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

#def discriminatorV1_model(leaky_alpha=0.2, dropRate=0.3, image_shape=(64,64,3)):
#     model = Sequential()
    
#    # layer1 (None,64,64,3)>>(None,32,32,32)
#    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
#    model.add(LeakyReLU(alpha=leaky_alpha))



def generator_model(nbrParamEntree=10, leaky_alpha=0.2):
    model = Sequential()
    
    # layer1 (None,500)>>(None,128*16*16)
    model.add(Dense(64*64*3, activation="sigmoid", input_shape=(nbrParamEntree,)))
    
    # (None,16*16*128)>>(None,16,16,128)
    model.add(Reshape((64, 64, 3)))
    
    # (None,16,16,128)>>(None,32,32,256)
    model.add(Conv2D(64, kernel_size=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    #(None,32,32,256)>>(None,32,32,256)
        
        
    #model.add(UpSampling2D())
    
    # (None,32,32,256)>>(None,32,32,256)
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # (None,32,32,256)>>(None,32,32,128)
    model.add(Conv2D(3, kernel_size=8, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("sigmoid"))
    
    #model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def discriminator_model(leaky_alpha=0.2, dropRate=0.3, image_shape=(32,32,3)):
    model = Sequential()
    
    # layer1 (None,64,64,3)>>(None,32,32,32)
    model.add(Conv2D(64, kernel_size=32, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(dropRate))
    # layer2 (None,32,32,32)>>(None,16,16,64)
    model.add(Conv2D(64, kernel_size=8, strides=2, padding="same"))
    # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=leaky_alpha))
    model.add(Dropout(dropRate))
    # (None,16,16,64)>>(None,8,8,128)
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
##    model.add(MaxPooling2D(pool_size=(2, 2)))
##    model.add(Dropout(0.25))
    # (None,8,8,64)
    model.add(Flatten())
##    model.add(Dense(64, activation='relu'))
##    model.add(BatchNormalization(momentum=0.8))
##    model.add(Dropout(dropRate))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(32, activation='relu'))
##    model.add(BatchNormalization(momentum=0.8))
##    model.add(Dropout(dropRate))
    model.add(Dense(1, activation='sigmoid'))
    #model.summary()
    sgd=SGD(lr=0.0002)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.5), metrics=['accuracy'])
    return model

def entrainement(epochs, nbrImageEntrainement, datasetImg, ChargeSauvegarde, epoch_start,pasEntrainement):
    #On crée le GAN
    (gan, g, d) = DCGAN(10,datasetImg)

    #On doit entraîner un peu le discriminateur
    d.trainable = True
    d.fit(datasetImg[0:nbrImageEntrainement],np.ones(nbrImageEntrainement),epochs=1)
    d.trainable = False

    if(ChargeSauvegarde == True):
        gan.load_weights('GAN.h5')
        g.load_weights('g.h5')
        d.load_weights('d.h5')

    dLossReal = []
    dLossGAN  = []
    gLossLogs = []

    #On init la variable de stockage d'image
    imgGenereNonCalib = np.ndarray(shape=(50, datasetImg.shape[1],datasetImg.shape[2],3),
                     dtype=np.float32)

    genImage = []

    for e in range(epochs):

        #On clear les images saves
        #genImage.clear()

        moyDlossR = 0.0
        moyDlossF = 0.0
        moyGAN = 0.0


        for a in range(0,nbrImageEntrainement,pasEntrainement):

            print("Epochs " + str(e+1) + "/" + str(epochs) + " image : " + str(a+1) + "/" + str(nbrImageEntrainement), end="\r")

            bruit = np.random.rand(pasEntrainement+1,10)
            #On génère l'image à partir du bruit
            genImage = g.predict(bruit)

            if(a < 50):
                imgGenereNonCalib = copy.copy(genImage)

            #On enregistre les 2 dernière images générées
            #if(a >= nbrImageEntrainement - 2):
            #    genImage.append(genImage)

            #On entraîne le discriminateur
            d.trainable = True
            dLossR = d.fit(datasetImg[a:a+pasEntrainement+1], np.ones(pasEntrainement+1),validation_split=0.2,verbose=0)
            dLossF = d.fit(genImage, np.zeros(pasEntrainement+1),validation_split=0.2,verbose=0)
            if(dLossR.history['acc'][0] == 0):
                moyDlossR = moyDlossR + 0
            else:
                moyDlossR = moyDlossR + (dLossR.history['acc'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))
            if(dLossF.history['acc'][0] == 0):
                moyDlossF = moyDlossF + 0
            else:
                moyDlossF = moyDlossF + (dLossF.history['acc'][0]/(nbrImageEntrainement/pasEntrainement))
            #dLoss = np.add(dLossF, dLossR) #* 0.5
            d.trainable = False
            #On entraîne le generateur
            gLoss = gan.fit(bruit, np.ones(pasEntrainement+1),validation_split=0.2,verbose=0)

            if(gLoss.history['acc'][0] == 0):
                moyGAN = moyGAN + 0
            else:
                moyGAN = moyGAN + (gLoss.history['acc'][0]/(float(nbrImageEntrainement)/float(pasEntrainement)))

        #On quantifie les résultats du GAN
        #dLossReal.append([e, dLoss[0]])
        #dLossFake.append([e, dLoss[1]])
        #gLossLogs.append([e, gLoss])
        
        dLossReal.append([e, moyDlossR])#dLossR.history['mean_absolute_percentage_error'][0]])
        #dLossFake.append([e, dLossF.history['mean_absolute_percentage_error'][1]])
        dLossGAN.append([e,moyGAN])
        gLossLogs.append([e, moyDlossF])#gLoss.history['mean_absolute_percentage_error'][0]])
        
        moyDlossR = 0.0
        moyDlossF = 0.0

        dLossRealArr = np.array(dLossReal)
        dLossGANArr  = np.array(dLossGAN)
        gLossLogsArr = np.array(gLossLogs)

        #Sauvegarde
        gan.save_weights('GAN.h5')
        g.save_weights('g.h5')
        d.save_weights('d.h5')

        #On enregistre les perfs du GAN
        Divers.SauvegardePerfGAN(e,dLossRealArr,dLossGANArr,gLossLogsArr,"Resultat")
       

        imgGenereNonCalib = copy.copy(genImage)
        imgGenereNonCalib = imgGenereNonCalib * 255.
        imgGenereNonCalib = imgGenereNonCalib.astype(np.uint8)

        #On enregistre l'image générée
        Divers.SauvegardeImageMatplot(5,2,imgGenereNonCalib,"Resultat/ImageGenerees/epochs_" + str(e) + ".png")

        imgGenereNonCalib = imgGenereNonCalib.astype(np.float32)
        #plt.figure(1)
        #plt.imshow(genImage[0,:,:,:])
        #plt.axis('off')
        #plt.savefig("Resultat/ImageGenerees/epochs_" + str(e+epoch_start) + ".png")
        #plt.close()



def afficheMeilleurImageGAN(nombreImg,nbrColonne,nbrLigne,nomFichier):

    bruit = np.random.rand(nombreImg+1,10)

    #On init la variable de stockage d'image
    dataset = np.ndarray(shape=(nbrColonne*nbrLigne, 64,64,3),
                     dtype=np.float32)

    #On crée le GAN
    (gan, g, d) = DCGAN(10,dataset)

    gan.load_weights('GAN.h5')
    g.load_weights('g.h5')
    d.load_weights('d.h5')

    imgListe = g.predict(bruit)
    pourcentageReussite = gan.predict(bruit)

    i=0
    index = 0
    while(i < ((nbrColonne*nbrLigne)-2) | index < nombreImg):
        print("Images trouvés : " + str(i) + " / " + str(nbrColonne*nbrLigne) + "   Image parcourue : " + str(index) + " / " + str(nombreImg), end="\r")
        
        if pourcentageReussite[index] > 0.7:
            if i < (nbrColonne*nbrLigne) :
                dataset[i] = imgListe[index]
            i+=1
        index+=1
    dataset = dataset * 255.
    dataset = dataset.astype(np.uint8)

    Divers.SauvegardeImageMatplot(nbrColonne,nbrLigne,dataset,"Resultat/"+nomFichier)



