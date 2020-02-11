import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli
import math

def mutar_individuo(I, p):
    IM = I[:]
    for i in range(len(I)):
        if np.random.random()<p:
            IM[i] = int(not I[i])
    return IM

def mutar_poblacion(poblacion,probabilidad_mutacion):
    poblacion_mutada=[]
    for i in range(len(poblacion)):
        poblacion_mutada.append(mutar_individuo(poblacion[i],probabilidad_mutacion))
    return poblacion_mutada

def funcion(x):
    return -(0.1+(1-x)**2-0.1*math.cos(6*math.pi*(1-x)))+2

def bin_decimal(num_binario):
    return (np.sum(num_binario)/len(num_binario))*2

tamanio_individuo = int(input("Tamanioo del individuo: "))
num_individuos = int(input("Numero de individuos: "))
num_generaciones = int(input("Numero de generaciones: " ))
probabilidad_mutacion = float(input("Probabilidad de mutacion: "))

# inicialización de la población
poblacion=[]
for i in range(num_individuos):
    poblacion+=[bernoulli.rvs(size=10,p=0.3)]
    
fx=[]
individuos=[]
for i in range(num_individuos):
    x=bin_decimal(poblacion[i])
    fx+=[funcion(x)]
    individuos+=[x]
plt.plot(individuos,fx,'x')

eje_y=[]
eje_x=np.arange(0,2,0.01)
plt.title("Generación 0")
for x in eje_x:
    eje_y+=[funcion(x)]
plt.plot(eje_x,eje_y)
plt.show()    


# evolución
promedio_fx=[]
bloques=num_generaciones//5
imprimir=bloques

for i in range(num_generaciones):
    fx = []
    individuos=[]
    #calcular fitness
    for j in range(num_individuos):
        x=bin_decimal(poblacion[j])
        fx+=[funcion(x)]
        individuos+=[x]
    if(i==imprimir):
        #gráfico de los resultados de la generación j
        plt.plot(individuos,fx,'x')

        eje_y=[]
        eje_x=np.arange(0,2,0.01)
        titulo="Generación: "+str(i)
        plt.title(titulo)
        for x in eje_x:
            eje_y+=[funcion(x)]
        plt.plot(eje_x,eje_y)
        plt.show()    
        imprimir+=bloques
    fitness=np.array(fx)/np.sum(fx)
    promedio_fx+=[np.mean(fx)]
    
    #cruzamiento
    offspring = []
    for i in range(num_individuos//2):
        parents = np.random.choice(num_individuos, 2, p=fitness)
        cross_point = np.random.randint(tamanio_individuo)
        offspring.append(list(poblacion[parents[0]][:tamanio_individuo]) + list(poblacion[parents[1]][tamanio_individuo:]))
        offspring.append(list(poblacion[parents[1]][:tamanio_individuo]) + list(poblacion[parents[0]][tamanio_individuo:]))
    poblacion = offspring
    
    #mutación
    poblacion=mutar_poblacion(poblacion,probabilidad_mutacion=probabilidad_mutacion)



fx=[]
individuos=[]
for i in range(num_individuos):
    x=bin_decimal(poblacion[i])
    fx+=[funcion(x)]
    individuos+=[x]
plt.plot(individuos,fx,'x')

eje_y=[]
eje_x=np.arange(0,2,0.01)
titulo="Generación "+str(num_generaciones)+" \nMejor individuo: "+str(individuos[np.argmax(fx)])
plt.title(titulo)
for x in eje_x:
    eje_y+=[funcion(x)]
plt.plot(eje_x,eje_y)
plt.show()  


plt.title("Promedio f(x)")
plt.plot(promedio_fx)
plt.show()