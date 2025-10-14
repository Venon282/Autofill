# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 08:33:07 2025

@author: PB267741
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
import os

from dotenv import load_dotenv
load_dotenv()  # Charge les variables depuis .env

PATH_DATABASE = os.getenv("PATH_DATABASE")

def fit_cyl_Ag(T,Longueur,wl):
    # Permet de fitter la transmission d'un cylindre d'argent dans de l'eau
    # Entrée:
        # T : Transmision (entre 0 et 1)
        # Longueur: Longueur du chemin optique utilisé en m
        # wl: longueur d'onde en nm (!!!!! Rester entre 300 et 1100 nm !!!!!!)
    # Sortie: 
        # D: Diamètre du cylindre en nm (entre 2 et 100 nm)
        # h: hauteur du cylindre en nm (entre 2 et 100 nm)
        # C: concentration des cylindre en particule/cm3
        # Tfit: Transmission fitté
        
    LD = np.arange(2,101) # Diamètres utilisés pour calculer la base de section effficace en nm
    Lh = np.arange(2,101) # Hauteurs utilisées pour calculer la base de section effficace en nm
    wl0 = np.arange(300,1100,10) # Longueur d'onde utilisées pour calculer la base de section effficace en nm
    MC = np.loadtxt(PATH_DATABASE) # Base de section efficace préalablement simulé

    T = PchipInterpolator(wl, T)(wl0)

    T[T <= 0] = 0
    T[T >= 1] = 1
    trust_region = T.astype(int)*0+1
    
    trust_region[T < 0.01] = 0
    trust_region[T > 0.99] = 0   
    trust_region[np.isnan(T)] = 0   
    trust_region[wl0 < 300] = 0
    trust_region[(wl0 > 1100) | (wl0 > wl[-1])] = 0

    trans_valid = int(np.sum(trust_region==1)/len(wl0)*100)
    #print('Transmission valide', trans_valid,'%')

    if trans_valid > 0:
        
        ## Calcul de B le produit concentration x longueur de chemin optique optimal dans le sens des moindre carré relatif
        B2 = np.atleast_2d(trust_region*1/np.log(T)**2)@(MC**2)
        B1 = np.atleast_2d(trust_region*1/np.log(T))@(MC)
        B = B1/B2

        indicB = np.atleast_2d(trust_region)@((np.atleast_2d(np.log(T)).transpose()-B*MC)/np.atleast_2d(np.log(T)).transpose())**2

        if np.all(np.isnan(indicB)):
            return None

        else:
            # ## Calcul des concentrations optimale
            C_n_B = -B[0,:]/Longueur# particule / m3

            # ## Calcul de la transmission fitté
            Tfit = np.exp(MC[:,indicB[0,:]==np.min(indicB[0,:])]*B[0,indicB[0,:]==np.min(indicB[0,:])])

            indicB[np.isnan(indicB)] = np.max(indicB[np.logical_not(np.isnan(indicB))])
            index_MC = np.argwhere(indicB[0,:]==np.min(indicB[0,:])).flatten()[0]

            # ## Calcul des paramètres de sortie
            indh = index_MC // len(LD)
            indD = index_MC % len(LD)

            D = LD[indD]
            h = Lh[indh]
            C = C_n_B[index_MC]*1E-6

            return D,h,C,Tfit
    
    else : 
        return None
