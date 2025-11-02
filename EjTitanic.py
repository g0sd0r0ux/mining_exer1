import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder;
from sklearn.decomposition import PCA;
from sklearn.impute import SimpleImputer;

# CONTEXTO
# Dataset "Titanic: Machine Learning from Disaster" de Kaggle, que contiene información sobre los pasajeros del Titanic, como su edad, clase, género, entre otros.

# PREGUNTA 1. Realizar carga y exploración de los datos, para ello cargaremos utilizando Pandas. Veremos:
#      ¿Cuántas filas y columnas tiene el dataset?
#      ¿Cuáles son los tipos de datos de cada columna?
# TIPS: Dataframe es similar a una hoja de excel, el r es para poder escribir una ruta completa

df = pd.read_csv(r'd:\practico\mineria\ejer1\train.csv');
print('RESOLUCIÓN 1');
print('*****************************');
print('Filas y columnas de los datos:\n', df.shape);
print('*****************************');
print('Tipos de datos de las columnas:\n', df.dtypes);
print('*****************************');
print('Primeras 5 filas:\n', df.head());
print('*****************************');

# PREGUNTA 2. Realizaremos detección de valores faltantes, Primero veremos si Hay valores faltantes en alguna columna.
#    Describiremos qué método usarías para tratarlos y justifica tu decisión.
print('RESOLUCIÓN 2');
print('*****************************');
valores_faltantes = df.isnull().sum();
print('Los valores faltantes son:\n', valores_faltantes);

# TIPS
# print('*****************************');
# print(df['Age'].head())
# --------
# Técnicas para datos faltantes con datos NUMÉRICOS
# Imputar con media (edad, precio, etc.)
# df['Age'].fillna(df['Age'].mean(), inplace=True)
# # Imputar con mediana (mejor si hay outliers)
# df['Age'].fillna(df['Age'].median(), inplace=True)
# # Imputar con valor específico
# df['Age'].fillna(0, inplace=True)
# --------
# Técnicas para datos faltantes con datos CATEGÓRICOS (strings)
# Imputar con moda (valor más frecuente)
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# # Imputar con valor específico
# df['Cabin'].fillna('Desconocido', inplace=True)
# # O eliminar si son muchos faltantes
# df.dropna(subset=['Embarked'], inplace=True)
# --------
# Imputación = Reemplazar valores faltantes con valores estimados
# --------
# CON inplace=True (modifica el DataFrame original)
# --------
# En Python los parámetros pueden ser: Obligatorios: fillna(valor). Opcionales: fillna(valor, inplace=True). Los opcionales tienen valores por defecto, por eso no siempre se especifican
# --------
# ¿Por qué "na" en fillna()? "na" viene de "Not Available" (No disponible). En estadística se usa NA para missing values. Pandas mantuvo esta convención: fillna(), isna(), dropna()
# --------
# Diferencia entre dropna() y drop()
# axis=1 = columnas, axis=0 = filas
# Eliminar FILAS con NaN en columna específica
# df.dropna(subset=['Embarked'])  # axis=0 por defecto
# # Eliminar COLUMNA completa
# df.drop('Cabin', axis=1)  # axis=1 = columnas
# # Eliminar FILAS con NaN en cualquier columna  
# df.dropna()  # sin subset = todas las columnas

# Para manejo de los datos faltanres en Age (númerica), usamos la mediana porque aún están los outliers
print('*****************************');
df['Age'].fillna(df['Age'].median(), inplace=True);
print('*****************************');
print('Los valores faltantes son:\n', valores_faltantes);
