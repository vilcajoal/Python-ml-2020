# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 02:48:46 2020

@author: vilcajoal
"""

# Plantilla de Pre Procesado

#IMPORTAR LIBRERÍAS
# Librerías Básicas
# numpy              : Contiene herramientas matemáticas
# matplotlib.pyplop  : Representación gráfica 
# pandas             : Carga y manipulación de datos 
import numpy as np                      
import matplotlib.pyplot as plt         
import pandas as pd                     

# IMPORTAR DATASET
# Sintaxis para cargar un csv con pandas
dataset = pd.read_csv("Data.csv")      
# iloc: sirve para localizar elementos por posición .iloc[filasinicio : filafinal, columnainicio : columnafinal]
# values: sirve para extraer el valor y no las posiciones.
X = dataset.iloc[:, :-1].values 
Y = dataset.iloc[:, 3].values  


# TRATAMIENTO DE NAN O VALORES NULOS
# importamos solo una parte de la librería con la siguietne sintaxis
from sklearn.preprocessing import Imputer 
# creando un objeto llamado imputer de la clase Imputer para manipulacion de nan
# parametro missing_values  : para saber los valores que deben ser detectados como desconocidos o nan
# parametro strategy        : se trata de la manera de reemplazar los valores nan en este caso se reemplaza por la media "mean"
# parametro axis            : es para indicar si se sustituye por la media de la fila o de la columna, si es fila "axis=1" y si es columna "axis=0"
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# metodo fit recibe un objeto y lo arregla "nan"
# En python cuando se pone 1:3 en realidad esta tomando valores desde el 1 hasta el 2 python no reconoce el ultimo elemento
# al igualar el imputer al imputer.fit lo que se hace es sobreescribir los valores modificados
# por ultimo se sobreescriben los valores en X 
# trasform se encarga de devolver y sustituir lo valores desconocidos a X
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#DATOS CATEGÓRICOS
#codificar datos categoricos, para eso importamos una nueva libreria para traducir 
#los paises a un numero diferente
from sklearn.preprocessing import LabelEncoder 
labelencoder_X = LabelEncoder() #codificador de datos categoritcos de X
# el .fit_transform() codifica las columnas que se le indican y se asignan a frame X
X[:,0] = labelencoder_X.fit_transform(X[:,0]) 
#Con esto se presenta un problema que ya que se le asigna datos 0, 1, 2 a las variables francia, españa, alemania y estas
#no son variables comparables ya que no son ordinales España no es mayor a Francia. Esto se a arregla con un concepto muy importante
#llamada variables Dummy que es una forma de traducir una variable categoritca(no tiene orden) en un conjunto detantas columnas como variables existen
#Francia = 1 0 0
#España  = 0 1 0
#Alemania= 0 0 0
# para esto utilizamos una segunda libreria para crear variables dummy
from sklearn.preprocessing import OneHotEncoder
onehotencoder_X = OneHotEncoder(categorical_features= [0])
X = onehotencoder_X.fit_transform(X).toarray()
# ahora creamos un labelencoder para la variable purshased no creamos una variable dummy ya que son variables booleanas "Yes" y "No"
labelencoder_Y = LabelEncoder()
Y= labelencoder_Y.fit_transform(Y) # Y es un vector por eso cambia la sintaxis

# DIVISION DEL DATASET EN CONJUNTO DE ENTRENAMIENTO Y TEST



















