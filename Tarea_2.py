import pandas as pd
from pathlib import Path
from CoolProp.CoolProp import PropsSI
import numpy as np
import matplotlib.pyplot as plt

# Obtiene la ruta exacta de la carpeta donde está este script (.py)
directorio_script = Path(__file__).parent

# Une esa ruta con el nombre de tu archivo CSV
ruta_csv = directorio_script / 'datos_estanque_v2.csv'

# Lee el archivo usando la ruta absoluta que acabamos de construir
df = pd.read_csv(ruta_csv, encoding='latin-1', sep=';', decimal=',')

# Previsualización de los datos
# print(df.head())

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
volumen_por_sensor = volumen_estanque / cantidad_sensores # volumen de agua por cada sensor en m3
altura_por_sensor = altura_estanque / cantidad_sensores # Altura por cada sensor en metros

# Propiedades de área
area_tapa = np.pi * (diametro_estanque / 2) ** 2  # Área de la base del estanque en m²
area_seccion_cilindro = np.pi * diametro_estanque * altura_por_sensor  # Área de la sección lateral del cada sección por sensor en m²

# Funciones de entalpías y entropías
def H_store(T_inicial_lista, T_actual_lista, V, P=101325):
    h_total_inicial = 0
    h_total_actual = 0
    for T_0, T_act in zip(T_inicial_lista, T_actual_lista):
        rho_j = PropsSI('D', 'T', T_act + 273.15, 'P', P, 'Water')
        h_j = PropsSI('H', 'T', T_act + 273.15, 'P', P, 'Water')

        rho_0 = PropsSI('D', 'T', T_0 + 273.15, 'P', P, 'Water')
        h_0 = PropsSI('H', 'T', T_0 + 273.15, 'P', P, 'Water')

        h_total_actual += V * rho_j * h_j
        h_total_inicial += V * rho_0 * h_0
    
    return h_total_actual - h_total_inicial
    
def S_store(T_inicial_lista, T_actual_lista, V, P=101325):
    s_total_inicial = 0
    s_total_actual = 0
    for T_0, T_act in zip(T_inicial_lista, T_actual_lista):
        rho_j = PropsSI('D', 'T', T_act + 273.15, 'P', P, 'Water')
        s_j = PropsSI('S', 'T', T_act + 273.15, 'P', P, 'Water')

        rho_0 = PropsSI('D', 'T', T_0 + 273.15, 'P', P, 'Water')
        s_0 = PropsSI('S', 'T', T_0 + 273.15, 'P', P, 'Water')

        s_total_actual += V * rho_j * s_j
        s_total_inicial += V * rho_0 * s_0
    
    return s_total_actual - s_total_inicial

def H_flow(T_in_1, T_in_2, T_out_1, T_out_2, V_dot_in1, V_dot_in2,  V_dot_out1, V_dot_out2, P=101325):
    rho_in1 = PropsSI('D', 'T', T_in_1 + 273.15, 'P', P, 'Water')
    m_dot_in1 = V_dot_in1 * rho_in1 / 1000 # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    h_in1 = PropsSI('H', 'T', T_in_1 + 273.15, 'P', P, 'Water')

    rho_in2 = PropsSI('D', 'T', T_in_2 + 273.15, 'P', P, 'Water')
    m_dot_in2 = V_dot_in2 * rho_in2 / 1000 # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    h_in2 = PropsSI('H', 'T', T_in_2 + 273.15, 'P', P, 'Water')

    rho_out1 = PropsSI('D', 'T', T_out_1 + 273.15, 'P', P, 'Water')
    m_dot_out1 = V_dot_out1 * rho_out1 / 1000  # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    h_out1 = PropsSI('H', 'T', T_out_1 + 273.15, 'P', P, 'Water')

    rho_out2 = PropsSI('D', 'T', T_out_2 + 273.15, 'P', P, 'Water')
    m_dot_out2 = V_dot_out2 * rho_out2 / 1000 # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    h_out2 = PropsSI('H', 'T', T_out_2 + 273.15, 'P', P, 'Water')

    return m_dot_in1 * h_in1 + m_dot_in2 * h_in2 - m_dot_out1 * h_out1 - m_dot_out2 * h_out2 # 1 minuto ya que los caudales están en L/min

def S_flow(T_in_1, T_in_2, T_out_1, T_out_2, V_dot_in1, V_dot_in2,  V_dot_out1, V_dot_out2, P=101325):
    rho_in1 = PropsSI('D', 'T', T_in_1 + 273.15, 'P', P, 'Water')
    m_dot_in1 = V_dot_in1 * rho_in1 / 1000 # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    s_in1 = PropsSI('S', 'T', T_in_1 + 273.15, 'P', P, 'Water')

    rho_in2 = PropsSI('D', 'T', T_in_2 + 273.15, 'P', P, 'Water')
    m_dot_in2 = V_dot_in2 * rho_in2 / 1000 # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    s_in2 = PropsSI('S', 'T', T_in_2 + 273.15, 'P', P, 'Water')

    rho_out1 = PropsSI('D', 'T', T_out_1 + 273.15, 'P', P, 'Water')
    m_dot_out1 = V_dot_out1 * rho_out1 / 1000  # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    s_out1 = PropsSI('S', 'T', T_out_1 + 273.15, 'P', P, 'Water')

    rho_out2 = PropsSI('D', 'T', T_out_2 + 273.15, 'P', P, 'Water')
    m_dot_out2 = V_dot_out2 * rho_out2 / 1000 # kg/min ya que V_dot es L/min y rho es kg/m³ y 1000 L = 1 m3
    s_out2 = PropsSI('S', 'T', T_out_2 + 273.15, 'P', P, 'Water')

    return m_dot_in1 * s_in1 + m_dot_in2 * s_in2 - m_dot_out1 * s_out1 - m_dot_out2 * s_out2 # 1 minuto ya que los caudales están en L/min

def H_hl_store(k, e, A_tapa, r_o, L, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_s7, t_s8, t_s9, t_s10, t_s11, t_inf):
    R_tapa = e / (k * A_tapa)
    R_centro = np.log(((r_o + e) / ( r_o ) )) / ( k * 2 * np.pi * L )
    UA_centro = 1/R_centro
    UA_tapa = UA_centro + 1/R_tapa

    h_hl_tapa = -( UA_tapa*(t_s1 - t_inf) + UA_tapa*(t_s11 - t_inf) ) 
    h_hl_centro = -( UA_centro*(t_s2 - t_inf) +
                     UA_centro*(t_s3 - t_inf) +
                     UA_centro*(t_s4 - t_inf) +
                     UA_centro*(t_s5 - t_inf) +
                     UA_centro*(t_s6 - t_inf) +
                     UA_centro*(t_s7 - t_inf) +
                     UA_centro*(t_s8 - t_inf) +
                     UA_centro*(t_s9 - t_inf) +
                     UA_centro*(t_s10 - t_inf)
                     )
    return ( h_hl_tapa + h_hl_centro ) * 60 # El 60 es el tiempo, y considera 1 minuto = 60 segundos ya que UA está en J /(s*K)
def S_hl_store(k, e, A_tapa, r_o, L, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_s7, t_s8, t_s9, t_s10, t_s11, t_inf):
    R_tapa = e / (k * A_tapa)
    R_centro = np.log(((r_o + e) / ( r_o ) )) / ( k * 2 * np.pi * L )
    UA_centro = 1/R_centro
    UA_tapa = UA_centro + 1/R_tapa

    s_hl_tapa = -( UA_tapa*(t_s1 - t_inf)/(t_s1 + 273.15) + UA_tapa*(t_s11 - t_inf)/(t_s11 + 273.15) ) 
    s_hl_centro = -( UA_centro*(t_s2 - t_inf)/(t_s2 + 273.15) +
                     UA_centro*(t_s3 - t_inf)/(t_s3 + 273.15) +
                     UA_centro*(t_s4 - t_inf)/(t_s4 + 273.15) +
                     UA_centro*(t_s5 - t_inf)/(t_s5 + 273.15) +
                     UA_centro*(t_s6 - t_inf)/(t_s6 + 273.15) +
                     UA_centro*(t_s7 - t_inf)/(t_s7 + 273.15) +
                     UA_centro*(t_s8 - t_inf)/(t_s8 + 273.15) +
                     UA_centro*(t_s9 - t_inf)/(t_s9 + 273.15) +
                     UA_centro*(t_s10 - t_inf)/(t_s10 + 273.15)
                     )
    return ( s_hl_tapa + s_hl_centro ) * 60 # El 60 es el tiempo, y considera 1 minuto = 60 segundos ya que UA está en J /(s*K)

f1 = df.iloc[0] # Fila 1



## Cálculos entalpías

df['H_store'] = df.apply(lambda r: H_store([ f1['TE1'], f1['TE2'], f1['TE3'], f1['TE4'], f1['TE5'], f1['TE6'],
                                            f1['TE7'], f1['TE8'], f1['TE9'], f1['TE10'], f1['TE11'] ],
                                            [ r['TE1'], r['TE2'], r['TE3'], r['TE4'], r['TE5'], r['TE6'],
                                            r['TE7'], r['TE8'], r['TE9'], r['TE10'], r['TE11'] ], volumen_por_sensor), axis=1 )

df['H_flow'] = df.apply(lambda r: H_flow(r['T36'], r['T51'], r['T35'], r['T52'], r['F32'], r['F51'], r['F31'], r['F51']), axis=1 )

df['H_hl_store'] = df.apply(lambda r: H_hl_store(k_1, espesor_1, area_tapa, diametro_estanque/2, altura_por_sensor, 
                                                 r['TE1'], r['TE2'], r['TE3'], r['TE4'], r['TE5'], r['TE6'],
                                                 r['TE7'], r['TE8'], r['TE9'], r['TE10'], r['TE11'], r['T_f']), axis=1 )



## Cálculos entropías

df['S_store'] = df.apply(lambda r: S_store([ f1['TE1'], f1['TE2'], f1['TE3'], f1['TE4'], f1['TE5'], f1['TE6'],
                                            f1['TE7'], f1['TE8'], f1['TE9'], f1['TE10'], f1['TE11'] ],
                                            [ r['TE1'], r['TE2'], r['TE3'], r['TE4'], r['TE5'], r['TE6'],
                                            r['TE7'], r['TE8'], r['TE9'], r['TE10'], r['TE11'] ], volumen_por_sensor), axis=1 )

df['S_flow'] = df.apply(lambda r: S_flow(r['T36'], r['T51'], r['T35'], r['T52'], r['F32'], r['F51'], r['F31'], r['F51']), axis=1 )

df['S_hl_store'] = df.apply(lambda r: S_hl_store(k_1, espesor_1, area_tapa, diametro_estanque/2, altura_por_sensor, 
                                                 r['TE1'], r['TE2'], r['TE3'], r['TE4'], r['TE5'], r['TE6'],
                                                 r['TE7'], r['TE8'], r['TE9'], r['TE10'], r['TE11'], r['T_f']), axis=1 )



## Acumulado de flujo (flow) y pérdidas de calor (hl) en entalpía y entropía

df['H_flow_acumulado'] = df['H_flow'].cumsum()
df['H_hl_store_acumulado'] = df['H_hl_store'].cumsum()

df['S_flow_acumulado'] = df['S_flow'].cumsum()
df['S_hl_store_acumulado'] = df['S_hl_store'].cumsum()

# ERROR y S_gen
df['Error_H'] = df['H_store'] - df['H_flow_acumulado'] + df['H_hl_store_acumulado']
df['S_gen'] = df['S_store'] - df['S_flow_acumulado'] - df['S_hl_store_acumulado']

# Convertimos el índice a horas (si cada fila es 1 minuto)
df['tiempo_hrs'] = df.index / 60 

# Convertimos a MegaJoules (MJ) para mejor escala [cite: 553]
df['H_store_MJ'] = df['H_store'] / 1e6
df['H_flow_acum_MJ'] = df['H_flow_acumulado'] / 1e6
df['H_hl_acum_MJ'] = df['H_hl_store_acumulado'] / 1e6

# Crear el gráfico
plt.figure(figsize=(10, 6))

# Ploteamos los componentes de la Primera Ley [cite: 550]
plt.plot(df['tiempo_hrs'], df['H_flow_acum_MJ'], 'k--', label='$\Delta H_{flow}^{exp}$')
plt.plot(df['tiempo_hrs'], df['H_store_MJ'], 'k-', label='$\Delta H_{store}^{exp}$')
plt.plot(df['tiempo_hrs'], df['H_hl_acum_MJ'], 'r-', linewidth=1, label='$\Delta H_{hl}^{exp}$')

# Configuración de formato profesional
plt.xlabel('Time [h]')
plt.ylabel('Energy [MJ]')
plt.title('Validación de la Primera Ley - Balance de Entalpías')
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

# (Opcional) Marcar las fases si conoces los tiempos de corte
# plt.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5) # Ejemplo: fin de carga

plt.tight_layout()
plt.show()


# 2. Escalamiento: Convertimos a kiloJoules por Kelvin (kJ/K)
df['S_store_kJ_K'] = df['S_store'] / 1e3
df['S_flow_acum_kJ_K'] = df['S_flow_acumulado'] / 1e3
df['S_hl_acum_kJ_K'] = df['S_hl_store_acumulado'] / 1e3
df['S_gen_kJ_K'] = df['S_gen'] / 1e3

# 3. Crear el gráfico de la Segunda Ley
plt.figure(figsize=(10, 6))

# Ploteamos los componentes de la Segunda Ley
plt.plot(df['tiempo_hrs'], df['S_flow_acum_kJ_K'], 'k--', label='$\Delta S_{flow}^{exp}$')
plt.plot(df['tiempo_hrs'], df['S_store_kJ_K'], 'k-', label='$\Delta S_{store}^{exp}$')
plt.plot(df['tiempo_hrs'], df['S_hl_acum_kJ_K'], 'r-', linewidth=1, label='$\Delta S_{hl}^{exp}$')

# Ploteamos la Entropía Generada (Irreversibilidades)
plt.plot(df['tiempo_hrs'], df['S_gen_kJ_K'], 'b-.', linewidth=1.5, label='$S_{gen}$ (Irreversibilidades)')

# Configuración de formato profesional
plt.xlabel('Time [h]')
plt.ylabel('Entropy [kJ/K]')
plt.title('Análisis de la Segunda Ley - Balance de Entropías')
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()