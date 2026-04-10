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
#print(df.head())


### Parámetros del estanque

# Diámetro del estanque en metros y Altura del estanque en metros
diametro_estanque = 1  # Diámetro del estanque en metros
altura_estanque = 2  # Altura del estanque en metros
cantidad_sensores = 11 # Cantidad de sensores de temperatura


### Propiedades Aislación

espesor_1 = 0.013 #https://www.comind.cl/producto/manta-epdm-aiscom-15m-x-1500mm-x-13mm/
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

# Propiedades agua
def H_agua(T, V):
    return V * (T + 273,15) * PropsSI('C', 'T', (T + 273,15), 'Q', 0, 'Water') * PropsSI('D', 'T', (T + 273,15), 'Q', 0, 'Water')

def H_flow(T,m):
    return m * (T + 273,15) * PropsSI('C', 'T', (T + 273,15), 'Q', 0, 'Water')








# Funciones para cálculos de números adimensionales y coeficientes de transferencia de calor por convección

def Gr(T_inf=float, T_sup=float, L_carac=float, fluid=str, g=9.80665):
    """Calcula el número de Grashof."""
    # Propiedades del fluido
    T_f = (T_inf + T_sup) / 2  # Temperatura de referencia para las propiedades del fluido
    mu = PropsSI('V', 'T', T_f, 'Q', 0, fluid)  # Viscosidad dinámica del fluido
    rho = PropsSI('D', 'T', T_f, 'Q', 0, fluid)  # Densidad del fluido
    beta = 1 / T_f  # Coeficiente de expansión volumétrica del fluido
    viscosidad_cinematica = mu / rho  # Viscosidad cinemática del fluido

    return (g * beta * abs(T_sup - T_f) * L_carac ** 3) / (viscosidad_cinematica ** 2)

def Pr(T_inf=float, T_sup=float, fluid=str):
    """Calcula el número de Prandtl."""
    # Propiedades del fluido
    T_f = (T_inf + T_sup) / 2  # Temperatura de referencia para las propiedades del fluido
    mu = PropsSI('V', 'T', T_f, 'Q', 0, fluid)  # Viscosidad dinámica del fluido
    k = PropsSI('L', 'T', T_f, 'Q', 0, fluid)  # Conductividad térmica del fluido
    c_p = PropsSI('C', 'T', T_f, 'Q', 0, fluid)  # Capacidad calorífica a presión constante del fluido

    return ( mu * c_p ) / k

def Ra(T_inf=float, T_sup=float, L_carac=float, fluid=str, g=9.80665):
    """Calcula el número de Rayleigh."""
    T_f = (T_inf + T_sup) / 2  # Temperatura de referencia para las propiedades del fluido
    mu = PropsSI('V', 'T', T_f, 'Q', 0, fluid)  # Viscosidad dinámica del fluido
    rho = PropsSI('D', 'T', T_f, 'Q', 0, fluid)  # Densidad del fluido
    k = PropsSI('L', 'T', T_f, 'Q', 0, fluid)  # Conductividad térmica del fluido
    c_p = PropsSI('C', 'T', T_f, 'Q', 0, fluid)  # Capacidad calorífica a presión constante del fluido
    beta = 1 / T_f  # Coeficiente de expansión volumétrica del fluido
    viscosidad_cinematica = mu / rho  # Viscosidad cinemática del fluido}

    return (g * beta * abs(T_sup - T_f) * L_carac ** 3 * mu * c_p) / (k * viscosidad_cinematica ** 2)

def Nu_sup_arriba(Ra=float, T_inf=float, T_sup=float):
    """Calcula el número de Nusselt para la superficie superior del estanque."""
    if T_inf > T_sup:
        if Ra > 1e5 and Ra < 1e11:
            return 0.27 * Ra ** (1/4)
        else:
            raise ValueError("El número de Rayleigh debe estar entre 1e5 y 1e11 para esta correlación.")
    else:
        if Ra > 1e4 and Ra < 1e7:
            return 0.54 * Ra ** (1/4)
        elif Ra > 1e7 and Ra < 1e11:
            return 0.15 * Ra ** (1/3)
        elif Ra > 1e8:
            return 0.14 * Ra ** (1/3)
        else:
            raise ValueError("El número de Rayleigh debe ser de al menos 1e4 para esta correlación.")

def Nu_sup_abajo(Ra=float, T_inf=float, T_sup=float):
    """Calcula el número de Nusselt para la superficie inferior del estanque."""
    if T_inf < T_sup:
        if Ra > 1e5 and Ra < 1e11:
            return 0.27 * Ra ** (1/4)
        else:
            raise ValueError("El número de Rayleigh debe estar entre 1e5 y 1e11 para esta correlación.")
    else:
        if Ra > 1e4 and Ra < 1e7:
            return 0.54 * Ra ** (1/4)
        elif Ra > 1e7 and Ra < 1e11:
            return 0.15 * Ra ** (1/3)
        elif Ra > 1e8:
            return 0.14 * Ra ** (1/3)
        else:
            raise ValueError("El número de Rayleigh debe ser de al menos 1e4 para esta correlación.")
        
def Nu_sup_vertical(Ra=float, Pr=float):
    """Calcula el número de Nusselt para la superficie vertical del estanque."""
    if Ra > 0.1 and Ra < 1e12:
        return ( 0.825 + 0.387 * Ra ** (1/6) / ( 1 + ( 0.492 / Pr ) ** (9/16) ) ** (8 / 27) ) ** 2
    else:
        raise ValueError("El número de Rayleigh debe estar entre 0.1 y 1e12 para esta correlación.")
    
def verificacion_cilindro_vertical(Gr=float, L=float, D=float):
    "Verifica si el cilindro se puede considerar como una placa vertical"
    if D >= 35*L/(Gr**(1/4)):
        return True
    else:
        return False
    
def h_conv(Nu=float, T_inf=float, T_sup=float, fluid=str, L=float):
    """Calcula el coeficiente de transferencia de calor por convección."""
    T_f = (T_inf + T_sup) / 2  # Temperatura de referencia para las propiedades del fluido
    k = PropsSI('L', 'T', T_f, 'Q', 0, fluid)  # Conductividad térmica del fluido
    return Nu * k / L

def resistencia_conveccion(h=float, A=float):
    """Calcula la resistencia térmica por convección."""
    return 1 / (h * A)

def resistencia_conduccion(k=float, L=float, A=float):
    """Calcula la resistencia térmica por conducción."""
    return L / (k * A)

def resistencia_en_paralelo(r_1, r_2):
    return (1/r_1 + 1/r_2) ** (-1)

def calor(T_inf, T_sup, resistencia):
    return abs(T_inf-T_sup)/resistencia


def tapa_superior(T_inf, T_sup, L, diametro, fluido):
    area_tapa = np.pi * (diametro / 2) ** 2
    area_ver = np.pi * diametro * L
    n_gr = Gr(T_inf, T_sup, L, fluido)
    n_pr = Pr(T_inf, T_sup, fluido)
    n_ra = n_gr * n_pr
    nu_arriba = Nu_sup_arriba(n_ra, T_inf, T_sup)
    if verificacion_cilindro_vertical(n_gr, L, diametro):
        nu_ver = Nu_sup_vertical(n_ra, n_pr)
    else:
        return "Error"
    h_arriba = h_conv(nu_arriba, T_inf, T_sup, fluido, L)
    h_ver = h_conv(nu_ver, T_inf, T_sup, fluido, L)
    r_arriba = resistencia_conveccion(h_arriba, area_tapa)
    r_ver = resistencia_conveccion(h_ver, area_ver)
    resistencia = resistencia_en_paralelo(r_arriba, r_ver)
    return calor(resistencia, T_inf, T_sup)

def tapa_inferior(T_inf, T_sup, L, diametro, fluido):
    area_tapa = np.pi * (diametro / 2) ** 2
    area_ver = np.pi * diametro * L
    n_gr = Gr(T_inf, T_sup, L, fluido)
    n_pr = Pr(T_inf, T_sup, fluido)
    n_ra = n_gr * n_pr
    nu_abajo = Nu_sup_abajo(n_ra, T_inf, T_sup)
    if verificacion_cilindro_vertical(n_gr, L, diametro):
        nu_ver = Nu_sup_vertical(n_ra, n_pr)
    else:
        return "Error"
    h_abajo = h_conv(nu_abajo, T_inf, T_sup, fluido, L)
    h_ver = h_conv(nu_ver, T_inf, T_sup, fluido, L)
    r_abajo = resistencia_conveccion(h_abajo, area_tapa)
    r_ver = resistencia_conveccion(h_ver, area_ver)
    resistencia = resistencia_en_paralelo(r_abajo, r_ver)
    return calor(resistencia, T_inf, T_sup)

def cilindro_exterior(T_inf, T_sup, L, diametro, fluido):
    area_ver = np.pi * diametro * L
    n_gr = Gr(T_inf, T_sup, L, fluido)
    n_pr = Pr(T_inf, T_sup, fluido)
    n_ra = n_gr * n_pr
    if verificacion_cilindro_vertical(n_gr, L, diametro):
        nu_ver = Nu_sup_vertical(n_ra, n_pr)
    else:
        return "Error"
    h_ver = h_conv(nu_ver, T_inf, T_sup, fluido, L)
    r_ver = resistencia_conveccion(h_ver, area_ver)
    return calor(r_ver, T_inf, T_sup)


