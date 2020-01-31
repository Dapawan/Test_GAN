#import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
from resizeimage import resizeimage

#Efface la console complète
#os.system('cls') 

def ChargementImages(nbrImg,chemin,largeur,hauteur):
    print("Chargement des " + str(nbrImg) + " images depuis le chemin " + chemin)

    #On récupère toutes les images du dossier
    cheminListe = glob.glob(chemin+"/*")

    #On init la variable de stockage d'image
    dataset = np.ndarray(shape=(nbrImg, hauteur,largeur,3),
                     dtype=np.uint8)

    for i in range(nbrImg):
        dataset[i] = Image.open(cheminListe[i])
        print("Image " + str(i) + " / " + str(nbrImg), end="\r")

    print("Fin du chargement des images")

    return dataset

def redimensionnementImage(nbrImg,chemin,largeur,hauteur):
    
    print("Demarrage du redimensionnement en " + str(largeur) + "*" + str(hauteur))

    #On init la variable de stockage d'image
    dataset_redimensionne = np.ndarray(shape=(nbrImg, hauteur,largeur,3),
                                        dtype=np.uint8)

    cheminListe = glob.glob(chemin+"/*")

    for i in range(nbrImg):
        print(str(i+1)+"/"+str(nbrImg), end="\r")


        tempImg = Image.open(cheminListe[i])

        #temp = np.array(tempImg)

        dataset_redimensionne[i] = resizeimage.resize_cover(tempImg, [largeur,hauteur])
        

    print("Fin du redimensionnement")

    return dataset_redimensionne

def sauvegardeImg(cheminAvecNom,img):
    #save
    imgpil = Image.fromarray(img)
    imgpil.save(cheminAvecNom + ".png")

def AffichageImageMatplot(nbrColonne,nbrLigne,Images):

    imageMatplot(nbrColonne,nbrLigne,Images)

    plt.show()

def SauvegardeImageMatplot(nbrColonne,nbrLigne,Images,cheminAvecNom):

    imageMatplot(nbrColonne,nbrLigne,Images)

    plt.savefig(cheminAvecNom)
    plt.close()

def imageMatplot(nbrColonne,nbrLigne,Images):

    fig, axs = plt.subplots(nbrLigne, nbrColonne)
    cnt = 0
    for i in range(nbrLigne):
        for j in range(nbrColonne):
            axs[i, j].imshow(Images[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1

    return plt

def SauvegardePerfGAN(epochs,dLossRealArr,dLossFakeArr,gLossLogsArr,chemin):

    plt.figure(1)
    plt.plot(dLossRealArr[:, 0], dLossRealArr[:, 1], label="Discriminator Loss - Real")
    #plt.plot(dLossFakeArr[:, 0], dLossFakeArr[:, 1], label="Discriminator Loss - Fake")
    plt.plot(gLossLogsArr[:, 0], gLossLogsArr[:, 1], label="Generator Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN')
    plt.grid(True)
    plt.savefig(chemin+'Result.png')
    plt.close()

def pixeliseAllImage(titre_dossier,imgs,nbrPixelLargeur,nbrPixelHauteur):
    print("Start pixelisation des ",imgs.shape[0]," images")
    for i in range(imgs.shape[0]):
        print("Image ",i+1,"/",imgs.shape[0], end="\r")
        imgpil = Image.fromarray(pixelisationImage(imgs[i],nbrPixelLargeur,nbrPixelHauteur))
        imgpil.save(titre_dossier+'/'+str(i)+".png")
    print("Fin pixelisation") 


def pixelisationImage(img,nbrPixelLargeur,nbrPixelHauteur):

    moyenneCouleur = np.ndarray(shape=(3),
                     dtype=np.float32)
    imagePixelisee = np.ndarray(shape=(img.shape[0],img.shape[1],img.shape[2]),
                     dtype=np.uint8)

    
    hauteur = 0
    largeur = 0
    compteurBloc = 0


    #On fait tout d'abord la moyenne de chaque bloc
    while(largeur+hauteur+nbrPixelHauteur+nbrPixelLargeur < img.shape[0]+img.shape[1] | compteurBloc != -1):
##        print("DEPART")
##        print("hauteur=",hauteur," ;largeur=",largeur)
        for y in range(hauteur,(hauteur+nbrPixelHauteur)):
            for x in range(largeur,(largeur+nbrPixelLargeur)):
                for indiceCouleur in range(3):
                    moyenneCouleur[indiceCouleur] += (1.0/float(nbrPixelLargeur*nbrPixelHauteur)) * float(img[x,y,indiceCouleur])
                    #moyenneCouleur[indiceCouleur] += (img[x,y,indiceCouleur])
        #1 bloc terminé -> on reproduit cette couleur sur un bloc dans l'image cible
##        print("ARRIVEE")
##        print("hauteur=",hauteur+nbrPixelHauteur," ;largeur=",largeur+nbrPixelLargeur)
        for y in range(hauteur,(hauteur+nbrPixelHauteur)):
            for x in range(largeur,(largeur+nbrPixelLargeur)):
                for indiceCouleur in range(3):
                    imagePixelisee[x,y,indiceCouleur] = (moyenneCouleur[indiceCouleur])

        moyenneCouleur[:] = 0
        largeur+=nbrPixelLargeur
        compteurBloc+=1
        #print(compteurBloc)
##        if(compteurBloc > 10):
##            break
        if(largeur+nbrPixelLargeur > img.shape[0]):
            largeur = 0
            hauteur+=nbrPixelHauteur
            if(hauteur+nbrPixelHauteur > img.shape[1]):
                compteurBloc = -1
            
        
    return imagePixelisee