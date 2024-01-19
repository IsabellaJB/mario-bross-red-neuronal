#  ------------ BIBLIOTECAS ------------
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# from funciones import convertir_a_gris

import torch.nn.functional as F
import PIL
from PIL import Image
import gym_super_mario_bros
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------------- ENTORNO VIRTUAL INCIO --------------------------------------------------------- 
# env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
# env = JoypadSpace(env, COMPLEX_MOVEMENT)
# n_output = len(COMPLEX_MOVEMENT)
# done = True


# #  --------------------------------------------------------- TOMAR EL FOTORAMA  --------------------------------------------------------- 
# env.reset()



def convertir_a_gris(rgb_matrix):
    img = Image.fromarray(rgb_matrix, 'RGB').convert('L')
    img_redimensionada = img.resize((50, 50)) # Cambiar tamaño 25 x 25 = 625
    return np.array(img_redimensionada)






def generar_V(poblacion, F):
    # Seleccionar tres índices aleatorios distintos
    r1, r2, r3 = np.random.choice(len(poblacion), 3, replace=False)
    # Generar el vector Donor (V)
    V = poblacion[r1] + F * (poblacion[r2] - poblacion[r3])
    return V

def generar_U(X, V, CR):
    # Generar el vector Trial (U) con crossover
    U = np.where(np.random.rand(len(X)) <= CR, V, X)
    return U










def calcular_V(r0, r1, r2, f):
    return r0 + f * (r1 - r2)

def calcular_U(V, gen_actual, CR):
    U = gen_actual.copy()
    for i in range(len(V)):
        valor_random = np.random.uniform(0, 1)
        if valor_random <= CR:
            U[i] = gen_actual[i] + CR * (V[i] - gen_actual[i])
    return U

def lista_pesos_a_dicc_pesos2(U_resultante):
    parte1 = np.array(U_resultante[:100000]).reshape(2500, 40) # Capa Entrada y la capa oculta 1
    parte2 = np.array(U_resultante[100000:100800]).reshape(40, 20) # Capa oculta 1 y la capa oculta 2
    parte3 = np.array(U_resultante[100800:]).reshape(20, 12) # Capa oculta 2 y la capa de salida

    diccionario_pesos = {
            'weights1': parte1,
            'weights2': parte2,
            'weights3': parte3
    }
    return diccionario_pesos



def lista_pesos_a_dicc_pesos(U_resultante):
    parte1 = U_resultante[:100000]
    parte2 = U_resultante[100000:100800]
    parte3 = U_resultante[100800:]

    diccionario_pesos = {
            'weights1': parte1.reshape(2500, 40),
            'weights2': parte2.reshape(40, 20),
            'weights3': parte3.reshape(20, 12)
    }
    return diccionario_pesos

    


def obs_Tensor(obs_gris):
    lista_obs_gris = np.array(obs_gris).flatten().tolist()
    tensor_obs_gris = torch.tensor(lista_obs_gris)
    tensor_obs_gris_float = tensor_obs_gris.to(dtype=torch.float32)
    dimensiones_actuales = tensor_obs_gris_float.size()
    tensor_obs_gris_float = tensor_obs_gris_float.view(1, -1) 

    return tensor_obs_gris


def primeros_10_cerebros(numero_Cerebros,df):
    # lista_reward = []
    lista_cont = []
    cont = 0

    for i in range(0,numero_Cerebros):    
        pesos_predefinidos = {
            'weights1': np.random.uniform(-2, 2, size=(2500, 40)),
            'weights2': np.random.uniform(-2, 2, size=(40, 20)),
            'weights3': np.random.uniform(-2, 2, size=(20, 12))
        }
        
        array1 = pesos_predefinidos['weights1']
        array2 = pesos_predefinidos['weights2']
        array3 = pesos_predefinidos['weights3']
        lista1 = array1.flatten().tolist()
        lista2 = array2.flatten().tolist()
        lista3 = array3.flatten().tolist()
        lista_completa = []
        lista_completa.extend(lista1)
        lista_completa.extend(lista2)
        lista_completa.extend(lista3)
        df.loc[cont] = lista_completa
        # lista_cont.append(cont)
        cont+=1
    # df['Recompensa'] = lista_reward
    # df['Cerebro'] = lista_cont

    return df



# def evaluar():
#     resultado_forward = mi_red_manual.forward(tensor_obs_gris_float, pesos_predefinidos)
#     accion_elegida = int(np.argmax(resultado_forward))
#     acciones_mario = COMPLEX_MOVEMENT
#     accion_final = acciones_mario[accion_elegida]
#     resultados_step = env.step(accion_elegida)
#     obs, reward, done, truncated, info = resultados_step

def mejor_fila_df(df):
    max_valor = df['Recompensa'].max()
    filas_max = df[df['Recompensa'] == max_valor]
    
    if len(filas_max) > 1:
        seleccion_aleatoria = np.random.choice(filas_max.index)
        fila_seleccionada = df.loc[seleccion_aleatoria]
        mejor_individuo = fila_seleccionada.to_numpy()
        if len(mejor_individuo) == 1:
            mejor_individuo = mejor_individuo[0]
    else:
        mejor_individuo = filas_max.to_numpy()[0] if len(filas_max) == 1 else filas_max.to_numpy()

    primeros_elementos = mejor_individuo[:-2]
    ultimos_dos_elementos = mejor_individuo[-2:]
    array_resul = np.array(primeros_elementos)

    return mejor_individuo, array_resul



def actualizar_mejor(df, mejor_individuo_anterior, array_anterior):
    mejor_individuo_nuevo, array_nuevo = mejor_fila_df(df)

    # NUEVO
    lista_nuevo = mejor_individuo_nuevo.tolist()  # Assuming mejor_individuo_nuevo is a NumPy array
    recompensa_nueva = lista_nuevo[-2]

    # ANTERIOR
    lista_anterior = mejor_individuo_anterior.tolist()  # Assuming mejor_individuo_anterior is a NumPy array
    recompensa_anterior = lista_anterior[-2]

    # print(recompensa_nueva)
    # print(recompensa_anterior)

    if recompensa_nueva >= recompensa_anterior:
        resul = recompensa_nueva
        # print(resul)
        mejor_fila = lista_nuevo
        mejor_array = array_nuevo
    else:
        resul = recompensa_anterior
        # print(resul)
        mejor_fila = lista_anterior
        mejor_array = array_anterior

    return mejor_fila, mejor_array, resul


    # resultado = 




    # return mejor_individuo, array_resul





def pesos_df_a_U(df,i):


    fila_original = df.iloc[i]

    df_sin_fila_original = df[df.index != fila_original.name]

    filas_aleatorias = df_sin_fila_original.sample(n=3, replace=False)

    arrays_filas_seleccionadas = filas_aleatorias.values
    r0 = arrays_filas_seleccionadas[0]
    r1 = arrays_filas_seleccionadas[1]
    r2 = arrays_filas_seleccionadas[2]

    f = 0.5  

    cross_over_rates = 0.8

    V_resultante = calcular_V(r0, r1, r2, f)
    V_resul = np.array(V_resultante)
    
        
    fila_array = np.array(fila_original)
    U_resultante = calcular_U(V_resul, fila_array, cross_over_rates)
    return U_resultante







def df_U_10primeras_Reward(lista_U, mi_red_manual,tensor_obs_gris_float,env):
    df_U_10primeras = pd.DataFrame(lista_U)
    lista_reward = []
    lista_cont = []
    cont = 0
    for i in range(0,len(lista_U)):
        diccionario_pesos = lista_pesos_a_dicc_pesos(lista_U[i])
        resultado_forward = mi_red_manual.forward(tensor_obs_gris_float, diccionario_pesos)
        accion_elegida = int(np.argmax(resultado_forward))
        acciones_mario = COMPLEX_MOVEMENT
        accion_final = acciones_mario[accion_elegida]
        # print(accion_elegida)
        resultados_step = env.step(accion_elegida)
        obs, reward, done, truncated, info = resultados_step
        lista_reward.append(reward)
        lista_cont.append(cont)
        cont+=1
    df_U_10primeras['Recompensa'] = lista_reward
    df_U_10primeras['Cerebro'] = lista_cont
    return df_U_10primeras




def sustituir_filas_df(mejores_recompensas, recompensa, df,lista_U):
    # lista U es una lista de listas
    for i in range(0,10):
        if mejores_recompensas[i] < recompensa[i]:
            nueva_fila = lista_U[i]
            indice_fila_a_reemplazar = i
            df.iloc[indice_fila_a_reemplazar] = nueva_fila
            mejores_recompensas[i] = recompensa[i]

    return df, mejores_recompensas






def mejor_fila_del_df(df, mejores_recompensas):
    otra_df = df.copy()
    otra_df['Recompensa'] = mejores_recompensas

    max_valor = otra_df['Recompensa'].max()
    filas_max = otra_df[otra_df['Recompensa'] == max_valor]

    if len(filas_max) > 1:
        seleccion_aleatoria = np.random.choice(filas_max.index)
        fila_seleccionada = otra_df.loc[seleccion_aleatoria]
        mejor_individuo = fila_seleccionada.to_numpy()
        if len(mejor_individuo) == 1:
            mejor_individuo = mejor_individuo[0]
    else:
        mejor_individuo = filas_max.to_numpy()[0] if len(filas_max) == 1 else filas_max.to_numpy()

    resul = mejor_individuo[:-1]
    array_resul = np.array(resul)

    return array_resul, mejor_individuo[-1]

