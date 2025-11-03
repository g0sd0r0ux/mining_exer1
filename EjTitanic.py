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


# 5. Detectaremos de valores atípicos, utilizaremos gráficos (boxplot) de Matplotlib, y pensaremos que podemos hacer con estos valores atípicos.
# --- Los valores atípicos son los datos que se alejan mucho del resto de los otros valores
# --- Se pueden detectar gráficamente o con algunas formulas, pero funciona de la misma manera
# --- boxplot usa o es equivalente al IQR
# --- IQR, es una manera en la que se determinan los valores atípicos por los cuartiles que le damos a los datos, normalmente se divide en 4
# --- Q1 = 0%-25%, Q2 = 25%-50%, Q3 = 50%-75%, Q4 = 75%-100%
# --- El IQR solo se queda con los datos de Q2 y Q3, ya que por lo general el 90% de los datos se encuentra en esos rangos cuando es una distribucción normal
# --- boxplot, muestra en un cuadrado los datos que están en la curva central y las rayas de de los costados y los outliers que se escapan de la distribucción normal
# --- EXPLICACIÓN GRÁFICO
# plt.subplot(1,2,1)           # ← Activo gráfico izquierdo
# sns.boxplot(y=df['Age'])     # ← Gráfico de Edad
# plt.title('Boxplot - Edad')  # ← Título para gráfico izquierdo
# plt.subplot(1,2,2)           # ← Activo gráfico derecho  
# sns.boxplot(y=df['Fare'])    # ← Gráfico de Fare
# plt.title('Boxplot - Fare')  # ← Título para gráfico derecho
# --- Lo que se ve en la imagen:
# Gráfico izquierdo (Age): La mayoría entre 20-40 años, algunos muy jóvenes/ancianos
# Gráfico derecho (Fare): Muchos outliers (precios de tickets muy altos)
# Rectángulo azul = Q1 a Q3 (50% datos centrales)
# Línea en el rectángulo = Mediana (valor del medio)
# Bigotes = Datos "normales" (fuera del 50% pero dentro del rango aceptable)
# Puntos fuera de bigotes = Outliers (valores atípicos)
# --- Sobre los outliers de edad (>55 años):
# -- ¿Mantenerlos o eliminarlos? Depende:
# OPCIÓN 1: MANTENER (RECOMENDADO para edad)
# - Personas mayores >55 años EXISTÍAN en el Titanic
# - Son datos reales, no errores
# - Proporcionan información valiosa
# OPCIÓN 2: ELIMINAR solo si:
# - Son errores de medición (ej: edad = 200)
# - Distorsionan mucho tu análisis específico

print('*****************************');
print('RESOLUCIÓN 5');

# Tamaño figura
plt.figure(figsize=(10,5));

# Gráfico edad a la izquierda
plt.subplot(1,2,1);
sns.boxplot(y=df['Age'])
plt.title('Boxplot - Edad');

# Gráfico tarifa a la derecha
plt.subplot(1, 2, 2);
sns.boxplot(y=df['Fare']);
plt.title('Boxplot - Fare');

# Se ajustan los gráficos y se muestra
plt.tight_layout();
plt.show();

# Identificar outliers por regla IQR (Fare)...
# IQR = Q3-Q1
# límite inferior = Q1 + (1.5 * IQR)
# límite superior = Q3 - (1.5 * IQR)

# Series = Una sola columna de un DataFrame
def detectar_outliers_iqr(series) :
    q1 = series.quantile(0.25);
    q3 = series.quantile(0.75);
    iqr = q3 - q1;
    print('*****************************');
    print('Valor de Q1', q1)
    print('*****************************');
    print('Valor de Q3: ', q3)
    print('*****************************');
    print('Valor de IQR: ', iqr)
    inferior = q1 - (1.5 * iqr);
    superior = q3 + (1.5 * iqr);
    print('*****************************');
    print('Valor inferior: ', inferior)
    print('Valor superior: ', superior)
    print('*****************************');
    print(series)
    return series[ (series < inferior) | (series > superior) ];

outliers_fare = detectar_outliers_iqr(df['Fare']);
print('*****************************');
print(f'Número de outliers (IQR) en Fare:  {outliers_fare.shape[0]}' );
print(outliers_fare.head());

# En ejemplo del Titanic:
# Fare = muchos outliers (precios muy altos de primera clase)
# Age = pocos outliers (personas muy ancianas)
# IQR es la forma MÁS COMÚN de detectar outliers numéricamente (boxplots lo usan internamente).


# 6. Realizaremos análisis de componentes principales, aplicando una técnica de reducción de dimensiones sobre las variables numéricas del dataset (por ejemplo, PCA).
# PCA (Principal Component Analysis)
# --- No todos los datos son analizables, pero si existe pueden existir correlaciones que no se deben perder, puediendo verse a través de la varianza
# --- Reducir dimenciones (menos columnas), pero que la correlación no se pierda (varianza)
# --- Objetivo transformar un conjunto de información correlacionado, en un conjunto más pequeño de variables que mantengan su varianza asociada
print('*****************************');
print('RESOLUCIÓN 6');

# Ejemplo para una variable de PCA
# Se piensan en variables que pueden estar relacionadas: altura, peso, ancho de hombros, talla de zapatos
# Una persona más alta es más probable que pese más, y sus hombros sean más anchos y tenga una talla de zapatos grandes (probablemente exista correlación)
# PCA se usa cuando hay muchas variables numéricas, las variables están correlacionadas, proyectar visualización más precisa sin tantas columnas, preprocesamiento de machile learning, comprensión de datos... (simplificando modelos)

# Variables que se van a analizar (Aplicar PCA) en base al contexto
num_feats=['Age', 'SibSp', 'Parch', 'Fare', 'Pclass'];
imputer = SimpleImputer(strategy='median');
X_num = imputer.fit_transform(df[num_feats]);

std = StandardScaler();
X_scaled = std.fit_transform(X_num);
pca = PCA(n_components=2);
X_pca = pca.fit_transform(X_scaled)

print("\n Varianza explicada por cada componente PCA", pca.explained_variance_ratio_)
print("Varianza explicada acumulada: ", np.cumsum(pca.explained_variance_ratio_))

plt.figure(figsize=(7,5));
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Survived'].astype(str), palette="deep")
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.title('PCA (2 componentes) - Titanic');
plt.legend(title='Survived');
plt.show();
