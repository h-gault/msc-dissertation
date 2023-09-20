import numpy as np
from scipy.integrate import quad
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from scipy.stats import norm
import time

start = time.time()


def init_(E0,I0,H0,R0,D0): #set initial conditions

    # Total population, N.
    N = 5000000

    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - E0 - H0 - D0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

    R_0 = 2.5*(E0/(R0+1))

    i = 3 #incubation
    h = (1/95)*(10000/(E0+1))
    x = 0
    r = 7/10 # recovery
    z = 1/10#receovery
    p = 0
    d = (1/1000)*(10000/(E0+1))
    f = 1/10
    k = 1/100

    params = i, h, x, r, z, p, d, f, k, R_0

    # A grid of time points (in days)
    t = np.linspace(0, 90, 90)

    # Initial conditions vector
    y0 = S0, E0, I0, H0, R0, D0

    return params, y0, N, t

def probabilities_func(E0_bounds,R0_bounds,distE,distR,E0s,R0s): #calculate probability for being in each E0/R0 range

    meanE, varE = distE
    meanR, varR = distR

    np.array(E0_bounds)
    np.array(R0_bounds)
    probE_ = norm.cdf(E0_bounds,meanE,varE)
    probR_ = norm.cdf(R0_bounds,meanR,varR)

    print(len(E0_bounds))
    print(len(probE_))

    E_probs = []
    R_probs = []
    for i in range(1,len(probE_)):
        E_probs.append(probE_[i]-probE_[i-1])
    for i in range(1,len(probR_)):
        R_probs.append(probR_[i]-probR_[i-1])

    dimE = len(E_probs)
    dimR = len(R_probs)
    print(dimE)
    #print(dimR)

    normE = sum(E_probs)
    normR = sum(R_probs)

    E_probs = E_probs/normE
    R_probs = R_probs/normR

    E_probs = np.array([E_probs])
    R_probs = np.array([R_probs])
    probs_matrix = np.matmul(E_probs.T,R_probs)

    print(E_probs)
    print(R_probs)
    print(probs_matrix)
    print(E_probs.sum())
    print(R_probs.sum())
    print(probs_matrix.sum())

    c = plt.imshow(probs_matrix, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Probability matrix')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show(block=False)
    plt.close()

    return E_probs, R_probs, probs_matrix


def boundaries(E0s, R0s): #calculate bounds for ranges

    E0_bounds = []
    R0_bounds = []

    E0_lower = (E0s[1]-E0s[0])/2
    E0_upper = (E0s[len(E0s)-1]-E0s[len(E0s)-2])/2
    R0_lower = (R0s[1]-R0s[0])/2
    R0_upper = (R0s[len(R0s)-1]-R0s[len(R0s)-2])/2

    if (E0s[0]- E0_lower) >0:
        E0_bounds.append(E0s[0]-E0_lower)
    else:
        E0_bounds.append(0)

    if (R0s[0]- R0_lower) >0:
        R0_bounds.append(R0s[0]-R0_lower)
    else:
        R0_bounds.append(0)

    for i in range(len(E0s)-1):
        E0_bounds.append(E0s[i]+(E0s[i+1]-E0s[i])/2)

    for i in range(len(R0s)-1):
        R0_bounds.append(R0s[i]+(R0s[i+1]-R0s[i])/2)

    E0_bounds.append(E0s[len(E0s)-1]+E0_upper)
    R0_bounds.append(R0s[len(R0s)-1]+R0_upper)

    print(E0_bounds)
    print(R0_bounds)

    return E0_bounds, R0_bounds

# The SIR model differential equations.
def deriv(y, t, N, i, h, x, r, z, p, d, f, k, R_0, Rld0, Rld1, Rld2):
    S, E, I, H, R, D = y

    if t>60:
        Rld = Rld2
    elif t>30:
        Rld = Rld1
    else:
        Rld = Rld0


    e =R_0*Rld*(d+h+r)

    dSdt = k*R -e*S*I/N
    dEdt = -i*E-x*E+e*I*S/N
    dIdt = i*E-d*I-h*I+p*H-r*I
    dHdt = h*I-p*H-f*H-z*H
    dRdt = z*H+r*I-k*R+x*E
    dDdt = d*I+f*H


    return dSdt, dEdt, dIdt, dHdt, dRdt, dDdt

def cost(H,D,I,Rld): #cost function including both hospital occupancy, deaths, and economic costs of lockdown
    Rld_cost =0
    for i in range(len(Rld)):
        Rld_cost = Rld_cost + 0.28*(1-np.sqrt(Rld[i]))*30793*5000000/365
    #print(D[90-1])
    #print(sum(H))
    #print(sum(Rld))
    c =sum(H)*2000+(0.5*sum(I)+sum(H))*30793/365+D[len(D)-1]*6000000

    return c+Rld_cost



def itera(params, y0, N,t,Rlds): #iterates over all possible lockdown combinations

    i, h, x, r, z, p, d, f, k, R_0 = params


    Rld_tuple = tuple(Rlds[1:4])
    Rld0, Rld1, Rld2 = Rld_tuple

    '''y0= list(y0)


    print(Rld0)

    y0.append(Rld0)

    y0 = tuple(y0)

    print(y0)'''

    # Integrate the SEIHRDS equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N,i, h, x, r, z, p, d, f, k, R_0, Rld0, Rld1,Rld2))
    #print(ret)
    S, E, I, H, R, D = ret.T

    Rld = []
    for i in range(30):
        Rld.append(Rld0)

    for i in range(30):
        Rld.append(Rld1)

    for i in range(30):
        Rld.append(Rld2)


    total_cost = cost(H,D,I,Rld)

    '''fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E/1000, 'g', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I/1000, 'y', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, H/1000, 'r', alpha=0.5, lw=2, label='In Hospital')
    ax.plot(t, R/1000, 'c', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, D/1000, 'k', alpha=0.5, lw=2, label='Dead')
    ax.plot(t, Rld, 'k', alpha=0.5, lw=2, ls='dashed', label='RLD')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', color='w', linewidth=2, linestyle='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show(block=False)
    plt.pause(3)
    plt.close()'''

    return S, E, I, H, R, D, Rld, total_cost

def run(start): #run main code

    dim = 15
    H0 = 200
    D0 = 100
    E0s = np.linspace(0,300000,dim)
    R0s = np.linspace(0,300000,dim)

    distE = (300000,300000)
    distR = (250000,300000)


    trials = (E0s, R0s)
    knowns = (H0,D0)

    E0_bounds, R0_bounds = boundaries(E0s,R0s)

    E_probs, R_probs, probs_matrix = probabilities_func(E0_bounds,R0_bounds,distE,distR,E0s,R0s)

    params, y0, N, t = init_(distE[0],distE[0],H0,distR[0],D0)

    poss_Rlds = [0.3,0.4,1.0]

    R_Array = []
    n = 1.
    for i in range(len(poss_Rlds)):
        for j in range(len(poss_Rlds)):
            for k in range(len(poss_Rlds)):
                R_Array.append([n,poss_Rlds[i],poss_Rlds[j],poss_Rlds[k]])
                n = n+1



    solutions_a = []
    min_inds = []
    for i in tqdm(range(len(probs_matrix))):
        solutions_a.append([])
        for j in range(len(probs_matrix[i])):
            costs = []
            solutions = []
            for Rlds in R_Array:
                params, y0, N, t = init_(E0s[i],E0s[i],H0,R0s[j],D0)
                S, E, I, H, R, D, Rld, total_cost = itera(params, y0, N,t,Rlds)
                #print(total_cost)
                costs.append(total_cost)
                solutions.append(Rlds[0])

            minimum = 10000000000000000000
            index = 0
            for cost in costs:
                if cost < minimum:
                    min_indexs = index
                    minimum = cost
                index = index + 1


            solutions_a[i].append(solutions[int(min_indexs)])
            min_inds.append(min_indexs)

    solutions_a = np.array(solutions_a)


    for i in range(len(solutions_a)):
        for j in range(len(solutions_a)):

            print(str(int(solutions_a[i,j])) + '&',end='')
        print('\\')
    print(solutions_a)
    min_inds = set(min_inds)
    min_inds = list(min_inds)

    print('The corresponding solutions are found below')
    for ind in min_inds:
        print(R_Array[ind])




    #print(R_Array)
    costs = []
    solutions = []
    for Rlds in tqdm(R_Array):
        expected_cost = 0
        for i in range(len(probs_matrix)):
            for j in range(len(probs_matrix[i])):
                params, y0, N, t = init_(E0s[i],E0s[i],H0,R0s[j],D0)
                S, E, I, H, R, D, Rld, total_cost = itera(params, y0, N,t,Rlds)
                #print(total_cost)
                expected_cost = expected_cost + probs_matrix[i][j]*total_cost

        costs.append(expected_cost)
        solutions.append([S, E, I, H, R, D, Rld])
        #if D.min() < 0:
            #print(Rlds)
            #print(D.min())
        #if H.min() <0:
            #print(Rlds)
            #print(H.min())
    #print(costs[1])

    #print(costs)
    minimum = 10000000000000000000
    index = 0
    for cost in costs:
        if cost < minimum:
            min_indexs = index
            minimum = cost
        index = index + 1


    print(min(costs))
    print(minimum)
    print(R_Array[min_indexs])

    minimum_sol = solutions[int(min_indexs)]

    end = time.time()

    elapsed_time = end - start

    print('run time')
    print(elapsed_time)

    S, E, I, H, R, D, Rld = tuple(minimum_sol)
    # Plot the data on three separate curves for S(t), I(t) and R(t)

    Rld_list = []
    for i in range(30):
        Rld_list.append(R_Array[min_indexs][1])
    for i in range(30):
        Rld_list.append(R_Array[min_indexs][2])
    for i in range(30):
        Rld_list.append(R_Array[min_indexs][3])

    fig, ax = plt.subplots()
    plt.yscale("log")
    ax.plot(t, S/1000000, 'b', lw=2, label='Susceptible')
    ax.plot(t, E/1000000, 'g', lw=2, label='Exposed')
    ax.plot(t, I/1000000, 'y', lw=2, label='Infected')
    ax.plot(t, H/1000000, 'r', lw=2, label='In Hospital')
    ax.plot(t, R/1000000, 'c', lw=2, label='Recovered with immunity')
    ax.plot(t, D/1000000, 'k', lw=2, label='Dead')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (Millions)')
    ax.set_ylim(0,5)
    secax = ax.secondary_yaxis('right', functions=(lambda x: x, lambda x: x))
    secax.set_ylabel('Lockdown Level')
    ax.plot(t, Rld, 'k', lw=2, ls='dashed', label='RLD')
    secax.set_ylim(0,1)
    ax.yaxis.set_tick_params(length=0)
    secax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    legend = ax.legend()
    plt.show()

    costs_matrix = []
    expected_cost = 0
    for i in range(len(probs_matrix)):
        costs_matrix.append([])
        for j in range(len(probs_matrix[i])):
            params, y0, N, t = init_(E0s[i],E0s[i],H0,R0s[j],D0)
            S, E, I, H, R, D, Rld, total_cost = itera(params, y0, N,t,R_Array[min_indexs])
            #print(total_cost)
            expected_cost = expected_cost + probs_matrix[i][j]*total_cost
            costs_matrix[i].append(total_cost)
    print('max')
    print(np.array(costs_matrix).max())
    c = plt.imshow(costs_matrix, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], norm = 'log', interpolation ='none')
    plt.colorbar(c, label='Cost (£)')
    plt.title('Cost of intervention ' + str(R_Array[min_indexs][1:])+ '\n for each starting condition')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()

    print(expected_cost)


run(start)
