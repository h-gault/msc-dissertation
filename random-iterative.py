import matplotlib
matplotlib.use('TKAgg')

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import csv
from tqdm import tqdm


waitstart = 100

#S=0
#E=1
#I=2
#H=3
#R=4
#D=5

def init():
    #take init conditions from csv input

    inpt= []
    # opening the CSV file
    with open('inputSEIRS.csv', mode ='r')as file:

        # reading the CSV file
        csvFile = csv.reader(file)
        next(csvFile)

                # displaying the contents of the CSV file
        for row in csvFile:
            inpt.append(row)


        input_ = np.array(inpt)

        row_no = int(input("which row?:"))

        size = int(input_[row_no][0])

        print('starting input: \n size: ' + str(size) + '\n S -> E: ' + input_[row_no][1] +'\n E -> I: ' + input_[row_no][2] + '\n I -> H: ' + input_[row_no][3] +'\n I -> R: ' + input_[row_no][4] +'\n I -> D: ' + input_[row_no][5] +'\n H -> D: ' + input_[row_no][6])

        probs = input_[row_no].astype(float)


        state=np.zeros((size,size),dtype=float)
        for i in range(0,size):
            for j in range(0,size):
                r = random.uniform(0,1)
                if r < (1/20):
                    state[i,j] = 1


        time=np.zeros((size,size),dtype=float)
        prior=np.zeros((size,size),dtype=float)

        frac = [[0],[0],[0],[0],[0],[0]]

    return state, size, probs, frac, time, prior


def update(c,l,probs,n,t,p,r): #updates state, requires the cell and a list list of 8 neighbours
    prob = l.count(2)
    if prob > 0:
        ni = 1
    else:
        ni = 0

    rand = random.uniform(0,1)


    e = 2.5*(probs[1]+probs[2]+probs[4])/8


#if currently susceptible
    if c == 0:
        if (ni == 1) & ((rand/(2.5-r)) < e):
            return 1
        else:
            return 0

#if currently exposed for three days or more
    elif (c == 1) & (t>3):
        if (rand < probs[0]) & (p==0):
            return 2
        elif (rand < (probs[0]*0.8)) & (p==1):
            return 2
        elif (rand < (probs[0]*0.6)) & (p>1):
            return 2
        elif (random.uniform(0,1) < (probs[0]*0.2)) & (p==1) :
            return 0
        elif (random.uniform(0,1) < (probs[0]*0.4)) & (p>1) :
            return 0
        else:
            return 1
#if currently infectious for five days or more
    elif (c == 2) & (t>5):
            if (rand < probs[1]) & (p==0):
                return 3
            elif (rand < (probs[1]*0.1)) & (p==1):
                return 3
            elif (rand < (probs[1]*0.01)) & (p>1):
                return 3
            elif random.uniform(0,1) < probs[2]:
                return 4
            elif (random.uniform(0,1) <probs[4]) & (p == 0):
                return 5
            else:
                return 2
#if currently in hospital
    elif c == 3:
            if rand < probs[3]:
                return 4
            elif random.uniform(0,1) < probs[5]:
                return 5

            else:
                return 3
#if currently resistant
    elif (c == 4) & (t > 60):
        if rand < probs[6]:
            return 0
        else:
            return 4
#if none of the above applies
    else:
        return c

def itera(state,probs,time,prior,n,r):

    time=time+1

    for i in range(size):
        for j in range(size):


            iAg=np.random.randint(0,size)
            jAg=np.random.randint(0,size)
            #print(state)
            c = state[iAg,jAg]
            t = time[iAg,jAg]
            p = prior[iAg,jAg]
            #take surrounding cells
            l = state[max(iAg-1,0):min(iAg+2,size), max(jAg-1,0):min(jAg+2,size)].flatten().tolist()

            state[iAg,jAg] = update(c,l,probs,n,t,p,r)

            #if state has changed in last iteration, reset state timer to 0
            if state[iAg,jAg] != c:
                time[iAg,jAg] = 0
                #if state is infectious, set prior infection to true
                if state[iAg,jAg] == 2:
                    prior[iAg,jAg] = prior[iAg,jAg]+1




    return state,time, prior #iterates across all states


def simulate(state,probs,frac,time,prior): #runs the simulation, produces graph

    n = 0



    rLD = []

    for i in tqdm(range(500)):
        r = 1.5
        rLD.append(r)
        n = n+1
        state, time, prior = itera(state,probs,time, prior,n,r)

        frac[0].append(np.sum(state == 0)/(size*size))
        frac[1].append(np.sum(state == 1)/(size*size))
        frac[2].append(np.sum(state == 2)/(size*size))
        frac[3].append(np.sum(state == 3)/(size*size))
        frac[4].append(np.sum(state == 4)/(size*size))
        frac[5].append(np.sum(state == 5)/(size*size))



    times = list(range(n+1))
    plt.plot(times,frac[0],label='Fraction Susceptible')
    plt.plot(times,frac[1],label='Fraction Exposed')
    plt.plot(times,frac[2],label='Fraction Infected')
    plt.plot(times,frac[3],label='Fraction in Hospital')
    plt.plot(times,frac[4],label='Fraction Resistant')
    plt.plot(times,frac[5],label='Fraction Dead')
    plt.xlabel('Time')
    plt.legend(loc='best')
    plt.show()




stateinit, size, probs, frac, time, prior = init()



simulate(stateinit,probs, frac,time,prior)
