# ------------ BIBLIOTECAS ------------
import funciones
import torch.nn.functional as F
import PIL
import gym_super_mario_bros
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim




from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from funciones import convertir_a_gris
from funciones import calcular_V
from funciones import calcular_U
from funciones import lista_pesos_a_dicc_pesos
from funciones import obs_Tensor
from funciones import primeros_10_cerebros
from funciones import mejor_fila_df
from funciones import pesos_df_a_U
from funciones import df_U_10primeras_Reward
from funciones import actualizar_mejor
from funciones import sustituir_filas_df
from funciones import mejor_fila_del_df

# from mario import evolucion_mario


from red_neronal import RedNeuronalManual


# Leer 'mis_pesos.txt'
finales_mejores_pesos = np.loadtxt('mis_pesos.txt')

# Leer 'mi_lista.txt'
finales_mejores_rewards = []
with open('mi_lista.txt', 'r') as archivo:
    for linea in archivo:
        finales_mejores_rewards.append(float(linea.strip()))




num_columnas = 101040
df = pd.DataFrame(finales_mejores_pesos,columns=range(num_columnas))
array_mejores_pesos, recompensa_del_mejor = mejor_fila_del_df(df, finales_mejores_rewards)



pesos_a_dic = lista_pesos_a_dicc_pesos(array_mejores_pesos)
# print(pesos_a_dic)
mejor_pesos = pesos_a_dic

print(mejor_pesos)


n_Entrada = 2500
n_Oculta1 = 40
n_Oculta2 = 20
n_Salida = len(COMPLEX_MOVEMENT)

mi_red_manual = RedNeuronalManual(n_Entrada, n_Oculta1, n_Oculta2, n_Salida)

# # # Configurar el entorno de Super Mario Bros
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

def evolucion_mario(n_Entrada,n_Oculta1,n_Oculta2,n_Salida,pesos_predefinidos):
    mi_red_manual = RedNeuronalManual(n_Entrada,n_Oculta1,n_Oculta2,n_Salida)
    recompensa = 0
    # done = True
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    for step in range(1000):
        obs_gris = convertir_a_gris(obs)
        tensor_obs_gris_float = obs_Tensor(obs_gris)
        resultado_forward = mi_red_manual.forward(tensor_obs_gris_float, pesos_predefinidos)
        accion_elegida = int(np.argmax(resultado_forward))
        resultados_step = env.step(accion_elegida)
        obs, reward, terminated, truncated, info = resultados_step
        # print("reward:", reward)
        recompensa += reward
        done = terminated or truncated
        if done:
            env.reset()
            recompensa = -1000
            # env.reset()
            # break        
    # return recompensa, obs
    return recompensa



evolucion_mario(n_Entrada,n_Oculta1,n_Oculta2,n_Salida,mejor_pesos)
