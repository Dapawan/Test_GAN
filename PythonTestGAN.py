import keras
import copy
import numpy as np
#Mes classes
import Divers
import GAN

#Variables
CheminDataset = 'data_trie'#'Dataset_pixellisation_all\Dataset_pixellisation_8_8'#'data_trie'#'data_redim_32x32'
nbrIMG = 501

#Les images peuvent être affichées entre 0 et 1 et aussi 0 et 255

#Chargement des images
Dataset = Divers.ChargementImages(nbrIMG,CheminDataset,64,64)

#Preparation du dataset pour entrainement
DatasetCalib = copy.copy(Dataset)

DatasetCalib.astype(np.float32)
DatasetCalib = DatasetCalib / 255.

# TESTS
#Divers.AffichageImageMatplot(5,5,Dataset[0:25])
#Divers.AffichageImageMatplot(5,5,DatasetCalib[0:25])
#Divers.SauvegardeImageMatplot(5,5,Dataset[0:25],"Resultat/ImageGenerees/Test")

#Dataset_redimensionne = Divers.redimensionnementImage(nbrIMG,CheminDataset,32,32)

#for i in range(nbrIMG):
#    Divers.sauvegardeImg("data_redim_32x32/"+str(i),Dataset_redimensionne[i])

GAN.entrainement(10000,500,DatasetCalib,False,0,50)
    #GAN.entrainement(50000,200,DatasetCalib,False,0,10)

#GAN.afficheMeilleurImageGAN(2000,10,10,"test")


#Divers.pixeliseAllImage('Dataset_pixellisation_all\Dataset_pixellisation_8_8',Dataset,8,8)
#Divers.pixeliseAllImage('Dataset_pixellisation_all\Dataset_pixellisation_4_4',Dataset,4,4)
#Divers.pixeliseAllImage('Dataset_pixellisation_all\Dataset_pixellisation_2_2',Dataset,2,2)
