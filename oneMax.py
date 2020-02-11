import random as rg
import numpy as np
import matplotlib.pyplot as plt


def bernoulli(p):
    if rg.random() < p:
        return 1  # Exito
    else:
        return 0  # Fracaso


def generateIndividual(p, size):
    I = []
    for i in range(size):
        I.append(bernoulli(p))

    return I


def init_population(p, size, individuals):
    P = []
    for i in range(individuals):
        P.append(generateIndividual(p, size))

    return P


def mutar_individuo(I, p):
    IM = I[:]
    for i in range(len(I)):
        if bernoulli(p):
            IM[i] = int(not I[i])

    return IM


def fitness_individuo(I):
    return float(sum(I)) / len(I)


def fitness_poblacion(P):
    FP = []
    for pi in P:
        FP.append(fitness_individuo(pi))

    return FP


def evaluate_poblacion(FP):
    EP = []
    for i in range(len(FP)):
        EP.append(float(FP[i]) / sum(FP))

    return EP


def commulative_fitness(EP):
    CF = [EP[0]]
    for i in range(1, len(EP)):
        CF.append(EP[i] + CF[i - 1])

    return CF


def rulette_selection(CF):
    ps = rg.random()
    seleccion = 0
    for pi in CF:
        if pi > ps:
            return seleccion

        seleccion += 1


def crossover(I1, I2, pc):
    return I1[:pc] + I2[pc:], I2[:pc] + I1[pc:]


def mean_population(Pob):
    prom = 0.0
    for i in Pob:
        prom += float(sum(i)) / len(i)

    return prom / len(Pob)


def main(args):
    """
    Uso: python agOneMax.py popSize indSize mutation steps
    popSize: size of the population
    indSize: size of the individuals
    mutations: mutation rate
    steps: evolution steps
    """
    if len(args) != 4:
        print(main.__doc__)
    else:
        popSize = int(args[0])
        indSize = int(args[1])
        mutation = float(args[2])
        steps = int(args[3])

        mP = []

        Pob = init_population(0.1, indSize, popSize)

        for t in range(steps):

            FP = fitness_poblacion(Pob)

            EP = evaluate_poblacion(FP)

            CF = commulative_fitness(EP)

            PN = Pob[:]
            for i in range(popSize // 2):
                inds = [rulette_selection(CF) for i in range(2)]
                pc = rg.randint(0, indSize)
                H1, H2 = crossover(Pob[inds[0]], Pob[inds[1]], pc)
                PN[inds[0]] = H1
                PN[inds[0]] = H2

            for pi in range(popSize):
                PN[pi] = mutar_individuo(PN[pi], mutation)

            Pob = PN[:]

            mP.append(mean_population(Pob))

        plt.plot(range(steps), mP)
        plt.show()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])