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


from red_neronal import RedNeuronalManual



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
            recompensa = -1000
            env.reset()
            # break        
    # return recompensa, obs
    return recompensa


# --------------------------------------------------------- ENTORNO VIRTUAL INCIO --------------------------------------------------------- 
env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, COMPLEX_MOVEMENT)

n_Entrada = 2500
n_Oculta1 = 40  
n_Oculta2 = 20 
n_Salida = len(COMPLEX_MOVEMENT)




# quedarme con la mejor generación de cerebros 


num_columnas = 101040
df = pd.DataFrame(columns=range(num_columnas))



numero_Cerebros = 10

df = primeros_10_cerebros(numero_Cerebros,df)
# print(df)



mejor_recompensa_global = -50

#poblacion incial
mejor_generacion = df.copy()

index = 0

mejores_pesos = mejor_generacion.iloc[0]

array_pesos = mejores_pesos.values

mejores_recompensas = [] 

numero_generaciones = 10


env.reset()
obs, reward, terminated, truncated, info = env.step(1)


for i in range(0,numero_generaciones):
    array = np.array(mejor_generacion.iloc[i])
    pesos_predefinidos = lista_pesos_a_dicc_pesos(array)
    resul = evolucion_mario(n_Entrada,n_Oculta1,n_Oculta2,n_Salida, pesos_predefinidos)
    mejores_recompensas.append(resul)




# numero_generaciones = 10
gen = 100
numero_generaciones = 10

array_mejores_pesos = None
recompensa = []

finales_mejores_pesos = []
finales_mejores_rewards = []



for j in range(0, gen):
    mejor_generacion = mejor_generacion.copy()
    recompensa = []
    lista_U = []
    obs = obs


    for i in range(0,numero_generaciones):
        u = pesos_df_a_U(mejor_generacion,i)

        pesos_predefinidos = lista_pesos_a_dicc_pesos(u)
        
        resul = evolucion_mario(n_Entrada,n_Oculta1,n_Oculta2,n_Salida, pesos_predefinidos)
        
        recompensa.append(resul)
        lista_U.append(u)
    



    mejor_generacion, mejores_recompensas = sustituir_filas_df(mejores_recompensas, recompensa,mejor_generacion,lista_U)
    array_mejores_pesos, recompensa_del_mejor = mejor_fila_del_df(mejor_generacion, mejores_recompensas)
    finales_mejores_pesos.append(array_mejores_pesos)
    finales_mejores_rewards.append(recompensa_del_mejor)

print(finales_mejores_pesos)
print(finales_mejores_rewards)
print(len(finales_mejores_pesos))
print(len(finales_mejores_rewards))
    
# print(recompensa)
np.savetxt('mis_pesos.txt', finales_mejores_pesos)
with open('mi_lista.txt', 'w') as archivo:
    for elemento in finales_mejores_rewards:
        archivo.write(str(elemento) + '\n')



