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
# CON inplace=True (modifica el DataFrame original), El warning es claro: df['Age'].fillna() con inplace=True ya no funciona bien en pandas modernas.
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
# --------
# Recomendación para todo el tratamiento:
# # Para Age (numérica)
# df['Age'] = df['Age'].fillna(df['Age'].median())
# # Para Embarked (categórica)
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# # Para Cabin (eliminar columna)
# df.drop('Cabin', axis=1, inplace=True)
# --------


# Para manejo de los datos faltantes en Age (númerica), usamos la mediana porque aún están los outliers
print('*****************************');
df['Age'] = df['Age'].fillna(df['Age'].median());
# Para el manejo de los datos faltanres en Embarked (categórico o string), usamos la moda porque no son muchos datos faltantes
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]);
# Para el manejo de los datos faltantes en Cabin (categórico o string), eliminamos la columna porque son muchos datos faltantes
df.drop('Cabin', axis=1, inplace=True);
valores_faltantes = df.isnull().sum();
print('Los valores faltantes son:\n', valores_faltantes);
print('*****************************');



# PREGUNTA 3. Haremos transformación de datos categóricos, utilizando la columna gender, codificandola en valores numéricos.

# TIPS 3
# --- ¿Limpieza o Transformación?
# - Limpieza: Arreglar problemas en los datos existentes
# Valores faltantes
# Outliers
# Duplicados
# Errores
# - Transformación: Cambiar el formato o estructura
# Codificación categórica
# Normalización
# Reducción dimensional
# --- Para columnas con MÁS categorías (Ej: Embarked)
# Si tuvieras que codificar Embarked (S, C, Q)
# df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# # O mejor:
# df = pd.get_dummies(df, columns=['Embarked'], prefix=['Embarked'])
# --- ¿Por qué transformar categóricos a numéricos?
# Los algoritmos de ML/matemáticos trabajan con números, no texto
# Funcionan mejor cuando los datos están en escalas similares (0-1, -1 a 1, etc.)
# Las operaciones matemáticas (sumas, multiplicaciones) requieren valores numéricos
# --- Técnicas de Trasformación de datos categóricos ¿Cuándo usar cada uno?
# - Usa Label Encoding cuando:
# La variable categórica SÍ tiene orden natural (Ej: "bajo", "medio", "alto")
# Tienes muchas categorías y One-Hot sería inmanejable
# Usas algoritmos que manejan bien variables ordinales (árboles de decisión)
# - Usa One-Hot Encoding cuando:
# La variable es nominal (sin orden natural) como "Sex", "Color", "País"
# Usas algoritmos basados en distancia (KNN, SVM, redes neuronales)
# Tienes pocas categorías (2-10)

# Transformación de datos categoricos en la columna Sex (male = 1, female = 0)
# --- Técnica Label Encoding:
# - Simple y directo :)
# - Crea orden artificial (1 > 0) cuando no debería haberlo :(
# print('*****************************');
# print('Antes:\n', df['Sex'].head());
# df['Sex'] = df['Sex'].map({'male': 1, 'female': 0});
# print('*****************************');
# print('Después:\n', df['Sex'].head());
# --- Técnica One-Hot Encoding (más usado):
# - Crea columnas separadas: Sex_male, Sex_female
# - Elimina orden artificial :)
# - Aumenta dimensionalidad :(
print('RESOLUCIÓN 3');
print('*****************************');
print('Antes:\n', df['Sex'].head());
df = pd.get_dummies(df, columns=['Sex'], prefix=['Sex'])
print('*****************************');
print('Después:\n', df['Sex_male'].head());
print('Después:\n', df['Sex_female'].head());
df = pd.get_dummies(df, columns=['Embarked'], prefix=['Embarked'])




# 4. Normalizaremos los datos, utilizando Min-Max Scaling (transformar datos entre 0 y 1). Mostraremos el nuevo rango de estos valores.
# x(normalizado) = ( X - X(min) ) / ( X(max) - X(min) )
# Edad: 1, 5, 6, 8, 4, 5, 4, 2, 3
# 3(normalizado) = (3-1) / (8-1) => 2 / 7 => 0.286
# --- El MinMaxScaler() permite transformar los datos pero conserva la distribución con una escala más pequeña para mayor precisión en en análisis de los datos con los respectivos algoritmos (distribucción de probabilidad de los datos).
# La normalización se realiza, para que sean valores que esten en un rango que sea mucho más sencillo para que los resultandos de los algoritmos sean más precisos.
# 4. Normalizaremos los datos, utilizando Min-Max Scaling (transformar datos entre 0 y 1). Mostraremos el nuevo rango de estos valores.
# --- Sintaxis [c for c in numeric_columns if c not in [...]]
# - Sí, es un "list comprehension" (como un for each compacto):
# FORMA LARGA:
# numeric_for_scaling = []
# for c in numeric_columns:
#     if c not in ["PassengerId", "Survived"]:
#         numeric_for_scaling.append(c)
# # FORMA CORTA (que usaste):
# numeric_for_scaling = [c for c in numeric_columns if c not in ["PassengerId", "Survived"]]
# --- ¿Por qué normalizar?
# Tu comprensión es correctísima:
# MinMaxScaler transforma: [valor_min, valor_max] → [0, 1]
# Objetivos:
# ✅ Estandarizar rangos: Todas las variables en misma escala (0-1)
# ✅ Mejorar algoritmos: SVM, KNN, redes neuronales funcionan mejor con datos escalados
# ✅ Evitar sesgo: Variables con rangos grandes no dominan el análisis
# ✅ Mantener distribución: Solo cambia la escala, no la forma de los datos


print('RESOLUCIÓN 4');
print('*****************************');

numeric_columns = df.select_dtypes(include=np.number).columns.to_list();
print(numeric_columns);
numeric_for_scaling = [c for c in numeric_columns if c not in ["PassengerId", "Survived"] ];
print(numeric_for_scaling);
print('*****************************');

scaler = MinMaxScaler();
df_scaled = df.copy();
print(df_scaled.head());
print('*****************************');
print(df_scaled.columns);
print('*****************************');
print(df_scaled.dtypes);
print('*****************************');
print(df_scaled.isnull().sum());


print('*****************************');
print('Edad antes:')
print(df_scaled['Age'].head());
print(df_scaled['Age'].agg(['min', 'max']));
df_scaled[numeric_for_scaling] = scaler.fit_transform(df_scaled[numeric_for_scaling])
print('*****************************');
print('Edad después:')
print(df_scaled['Age'].head());
print(df_scaled['Age'].agg(['min', 'max']));

