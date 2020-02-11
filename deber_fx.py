import matplotlib.pyplot as plt
import numpy as np
import math

def mutate(individuals, prob, pool):
    for i in range(len(individuals)):
        mutate_individual=individuals[i]
        if np.random.random() < prob:
            mutation = np.random.choice(pool[0])
            mutate_individual = [mutation] + mutate_individual[1:]
        
        for j in range(1,len(mutate_individual)):
            if np.random.random() < prob:
                mutation = np.random.choice(pool[1])
                mutate_individual = mutate_individual[0:j] + [mutation] + mutate_individual[j+1:]
        individuals[i] = mutate_individual
        
def fx(x):
    return -(0.1+(1-x)**2-0.1*math.cos(6*math.pi*(1-x)))+2

def decimalToList(num):
    genes=list(str(num))
    genes.pop(1)        
    return genes

def listToDecimal(num):
    decimal=0
    for i in range(len(num)):
        decimal+=num[i]*10**(-i)
    return decimal


def main(args):
    """
    Use: 
    python nombre.py ind_size n_ind n_gen p_mut
    Parametros:
    ind_size: size of individuals
    n_ind: number of individuals
    n_gen: number of generations
    p_mut: probability of mutation
    Ejemplo:
    python ga.py 16 100 100 0.00025
    """    
    if len(args) == 4:
        ind_size = int(args[0]) #size de los individuos
        n_ind = int(args[1]) #número de individuos
        n_gen = int(args[2]) #número de generaciones
        p_mut = float(args[3]) #probabilidad de mutación

        #Genetic pool
        genetic_pool=[[0,1],[0,1,2,3,4,5,6,7,8,9]]

        # First generation
        population=[]
        for i in range(n_ind):
            individuo=[]
            individuo += [np.random.choice(genetic_pool[0])]
            for j in range(ind_size-1):
                individuo += [np.random.choice(genetic_pool[1])]
            population+=[individuo]

        #Plot
        plt.subplot(2,2,1)
        y_axis=[]
        x_axis=[]
        for i in range(n_ind):
            x=listToDecimal(population[i])
            y_axis+=[fx(x)]
            x_axis+=[x]
        plt.plot(x_axis,y_axis,'x')

        y=[]
        x=np.arange(0,2,0.02)
        plt.title("First generation")

        for num in x:
            y+=[fx(num)]
        plt.plot(x,y)
            
        # Evolve for n_gen generations
        mean_fx=[]

        for gi in range(n_gen):
            fitness = []
            x_axis=[]
            for i in range(n_ind):
                x=listToDecimal(population[i])
                y=fx(x)
                fitness+=[y]
                x_axis+=[x]
            if(gi==(n_gen//2)):
                plt.subplot(2,2,2)
                plt.plot(x_axis,fitness,'x')
                y=[]
                x=np.arange(0,2,0.02)
                plt.title("Middle generation")

                for num in x:
                    y+=[fx(num)]
                plt.plot(x,y)                
            fitness=np.array(fitness)
            mean_fx.append(fitness.mean())
            fitness=fitness/fitness.sum()
            offspring = []
            for i in range(n_ind//2):
                parents = np.random.choice(n_ind, 2, p=fitness)
                cross_point = np.random.randint(ind_size)
                offspring += [population[parents[0]][:cross_point] + population[parents[1]][cross_point:]]
                offspring += [population[parents[1]][:cross_point] + population[parents[0]][cross_point:]]
            population = offspring
            mutate(population,p_mut,genetic_pool)

        plt.subplot(2,2,4)

        y_axis=[]
        x_axis=[]
        for i in range(n_ind):
            x=listToDecimal(population[i])
            y_axis+=[fx(x)]
            x_axis+=[x]
        plt.plot(x_axis,y_axis,'x')
        y_axis=np.array(y_axis)
        string="Mejor individuo:"+str(x_axis[y_axis.argmax()])
        plt.suptitle(string)
        y=[]
        x=np.arange(0,2,0.02)
        plt.title("Final generation")

        for num in x:
            y+=[fx(num)]
        plt.plot(x,y)


        plt.subplot(2,2,3)
        plt.title("Mean fx")
        plt.plot(mean_fx)
        plt.show()
    else:
        print(main.__doc__)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

