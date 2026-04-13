import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI
from CoolProp.HumidAirProp import HAPropsSI
import fluids as fld
import numpy as np

from pint import UnitRegistry
from fluids.units import *
from scipy.constants import g
import matplotlib.pyplot as plt

# Obtiene la ruta exacta de la carpeta donde está este script (.py)
directorio_script = Path(__file__).parent

# Une esa ruta con el nombre de tu archivo CSV
ruta_csv = directorio_script / 'datos_estanque_v2.csv'

# Lee el archivo usando la ruta absoluta que acabamos de construir
df = pd.read_csv(ruta_csv, encoding='latin-1', sep=';', decimal=',')

# Previsualización de los datos
print(df.head())


### Parámetros del estanque

# Diámetro del estanque en metros y Altura del estanque en metros
diametro_estanque = 1  # Diámetro del estanque en metros
altura_estanque = 2  # Altura del estanque en metros
cantidad_sensores = 11 # Cantidad de sensores de temperatura


### Propiedades Aislación

espesor_1 = 0.013 # https://www.comind.cl/producto/manta-epdm-aiscom-15m-x-1500mm-x-13mm/
k_1 = 0.038 # W/(m * °C)

### Propiedades del estanque

# Volumen del estanque en metros cúbicos
volumen_estanque = np.pi * (diametro_estanque / 2) ** 2 * altura_estanque

# Propiedades por sensor
volumen_por_sensor = volumen_estanque / cantidad_sensores # volumen de agua por cada sensor
altura_por_sensor = altura_estanque / cantidad_sensores # Altura por cada sensor

# Propiedades de área
area_tapa = np.pi * (diametro_estanque / 2) ** 2  # Área de la base del estanque en m²
area_seccion_cilindro = np.pi * diametro_estanque * altura_por_sensor  # Área de la sección lateral del cada sección por sensor en m²

# Función para calcular UA coeficiente global de transferencia de calor
def UA(k, e, A_tapa, r_o, L):
    R_centro = e / (k * A_tapa)
    R_tapa = np.log(((r_o + e) / ( r_o ) )) / ( k * 2 * np.pi * L )
    return [1/R_centro, 1 / (R_centro + R_tapa)]

# Cálculo de entalpías
def H_store(T_1, T_2, V):
    return V * ((T_1 + 273.15) * PropsSI('C', 'T', (T_1 + 273.15), 'Q', 0, 'Water') * PropsSI('D', 'T', (T_1 + 273.15), 'Q', 0, 'Water') - (T_2 + 273.15) * PropsSI('C', 'T', (T_2 + 273.15), 'Q', 0, 'Water') * PropsSI('D', 'T', (T_2 + 273.15), 'Q', 0, 'Water'))

def H_dot_flow(T_1, T_2, V_dot):
    return V_dot * ((T_1 + 273.15) * PropsSI('C', 'T', (T_1 + 273.15), 'Q', 0, 'Water') * PropsSI('D', 'T', (T_1 + 273.15), 'Q', 0, 'Water') - (T_2 + 273.15) * PropsSI('C', 'T', (T_2 + 273.15), 'Q', 0, 'Water') * PropsSI('D', 'T', (T_2 + 273.15), 'Q', 0, 'Water'))

def H_hl_store()
primera_fila = df.iloc[0]

