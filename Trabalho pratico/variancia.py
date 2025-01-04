import pandas as pd
import numpy as np

# Carregando os dados
df = pd.read_csv('heart-disease.csv')

# Calculando a variância para cada atributo
variancias = df.var()
print(variancias)