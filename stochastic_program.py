import numpy as np
from scipy.integrate import quad
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from scipy.stats import norm
import time

start = time.time()

def init_(E0, I0, H0, R0, D0): # set initial values

    # Total population, N.
    N = 5000000
    # Initial number of infected and recovered individuals, inputed via init_
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - E0 - H0 - D0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

    R_0 = 2.5*(E0/(R0+1))

    i = 3 #incubation
    h = (1/95)*(10000/(E0+1))
    #print(h)
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

def probabilities_func(E0_bounds,R0_bounds,distE,distR,E0s,R0s): #calculate E0, R0 probability distribution

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
    plt.show()

    return E_probs, R_probs, probs_matrix


def boundaries(E0s, R0s): #calculate boundaries for E0, R0 intervals

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

# The SIERHDS model differential equations
def deriv(y, t, N, i, h, x, r, z, p, d, f, k, R_0, Rld0, Rld1, Rld2): #differential equation for specified set of three lockdowns
    S, E, I, H, R, D = y


    if t>60:
        Rld = Rld2
    elif t>30:
        Rld = Rld1
    else:
        Rld = Rld0


    e =R_0*Rld*(d+h+r) #calcualte e according to Rld

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

    return c

def cost_ld(Rld): # just the econonmic costs of lockdown
    Rld_cost =0
    for i in range(len(Rld)):
        Rld_cost = Rld_cost + 0.28*(1-np.sqrt(Rld[i]))*30793*5000000/365
    return 30*Rld_cost

def cost_HD(H,D,I): # just the costs of hospital occupancy, infections, and deaths
    return sum(H)*2000+(0.5*sum(I)+sum(H))*30793/365+D[len(D)-1]*6000000


def itera(params, y0, N,t,Rlds): #solves the differential equation for a specified set of three lockdown levels, calculates the costs at each period.

    i, h, x, r, z, p, d, f, k, R_0 = params

    if len(Rlds) == 3:

        Rld_tuple = tuple(Rlds)
        #print(Rld_tuple)
        Rld0, Rld1, Rld2 = Rld_tuple

        # Integrate the SEIHRDS equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N,i, h, x, r, z, p, d, f, k, R_0, Rld0, Rld1,Rld2))

    else:
        print('Error - ensure lockdown list is the correct length')

    #print(ret)
    S, E, I, H, R, D = ret.T

    cost_full = cost(H,D,I,Rlds) #and for p1/p2a aswell
    costHD_full = cost_HD(H,D,I)

    costHD_p1 = cost_HD(H[30:],D[30:], I[30:])
    costHD_p2 = cost_HD(H[60:],D[60:], I[30:])

    Rld_list = []
    for i in range(30):
        Rld_list.append(Rld0)
    for i in range(30):
        Rld_list.append(Rld1)
    for i in range(30):
        Rld_list.append(Rld2)

    '''
    if max(E) > 2000:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        ax.plot(t, S/1000000, 'b', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(t, E/1000000, 'g', alpha=0.5, lw=2, label='Exposed')
        ax.plot(t, I/1000000, 'y', alpha=0.5, lw=2, label='Infected')
        ax.plot(t, H/1000000, 'r', alpha=0.5, lw=2, label='In Hospital')
        ax.plot(t, R/1000000, 'c', alpha=0.5, lw=2, label='Recovered with immunity')
        ax.plot(t, D/1000000, 'k', alpha=0.5, lw=2, label='Dead')
        ax.plot(t, Rld_list, 'k', alpha=0.5, lw=2, ls='dashed', label='RLD')
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Number (Millions)')
        ax.set_ylim(0,max(S)*1.05/1000000)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(True, which='major', color='w', linewidth=2, linestyle='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    '''
    return S, E, I, H, R, D, cost_full, costHD_full, costHD_p1, costHD_p2


def run_p1(params, y0, N, t, Rld0 ): #finds the minimum cost, where only the period 1 lockdown level is specified (periods 2 and 3 are free choice)


    '''
    Rlst = []
    with open('inputs4_0.csv', mode = 'r') as file:

        Rcombos = csv.reader(file)
        next(Rcombos)

        for row in Rcombos:
            Rlst.append(row)

    R_Array = np.array(Rlst).astype(float)
    '''
    poss_Rlds = [0.3,0.4,1.0]

    m = 0
    for i in range(len(poss_Rlds)):
        if poss_Rlds[i] == Rld0:
            m=i
            break
    R_Array = []
    n = 9*m+1
    for i in range(len(poss_Rlds)):
        for j in range(len(poss_Rlds)):
            R_Array.append([n,Rld0,poss_Rlds[i],poss_Rlds[j]])
            n = n+1
    #print(R_Array)
    costs_full = []
    costs_HD_full = []
    costs_HD_p1 = []
    costs_HD_p2 = []
    solutions = []
    for Rlds in R_Array:
        S, E, I, H, R, D, cost_full, costHD_full, costHD_p1, costHD_p2 = itera(params, y0, N,t,Rlds[1:])
        costs_full.append(cost_full)
        costs_HD_full.append(costHD_full)
        costs_HD_p1.append(costHD_p1)
        costs_HD_p2.append(costHD_p2)
        solutions.append([S, E, I, H, R, D, Rlds])
        #if D.min() < 0:
            #print(Rlds)
            #print(D.min())
        #if H.min() <0:
            #print(Rlds)
            #print(H.mi #finds thn())
    #print(costs[1])

    #print(costs_full)
    minimum_full = 1000000000000000000000000000
    index = 0
    for cost in costs_full:
        if cost < minimum_full:
            min_indexs = index
            minimum_full = cost
        index = index + 1

    #print(minimum_full)

    min_HD = costs_HD_full[min_indexs]
    min_HD_p1 = costs_HD_p1[min_indexs]
    min_HD_p2 = costs_HD_p2[min_indexs]
    #print(min(costs))
    #print(minimum)
    #print(R_Array[min_indexs])

    minimum_sol = tuple(solutions[int(min_indexs)])
    #print('minimum_sol')
    #print(minimum_sol)
    S, E, I, H, R, D, Rlds = minimum_sol

    Rld = Rlds[:len(t)]

    '''

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E/1000000, 'g', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I/1000000, 'y', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, H/1000000, 'r', alpha=0.5, lw=2, label='In Hospital')
    ax.plot(t, R/1000000, 'c', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, D/1000000, 'k', alpha=0.5, lw=2, label='Dead')
    ax.plot(t, Rld, 'k', alpha=0.5, lw=2, ls='dashed', label='RLD')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (millions)')
    ax.set_ylim(0,5)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', color='w', linewidth=2, linestyle='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.show(block=False)
    #plt.pause(1)
    plt.close()
    '''

    return R_Array[min_indexs], minimum_full, min_HD, min_HD_p1, min_HD_p2, H, D

def run_p2(params, y0, N, t, Rld0, Rld1): #finds the minimum cost, where the period 1 and 2  lockdow levels are specified (period 3 is free choice)


    '''
    Rlst = []
    with open('inputs4_0.csv', mode = 'r') as file:

        Rcombos = csv.reader(file)
        next(Rcombos)

        for row in Rcombos:
            Rlst.append(row)

    R_Array = np.array(Rlst).astype(float)
    '''
    poss_Rlds = [0.3,0.4,1.0]

    m = 0
    for i in range(len(poss_Rlds)):
        if poss_Rlds[i] == Rld0:
            m=i
            break
    p = 0
    for i in range(len(poss_Rlds)):
        if poss_Rlds[i] == Rld1:
            p=i
            break
    R_Array = []
    n = 9*m+3*p+1
    for i in range(len(poss_Rlds)):
        R_Array.append([n,Rld0, Rld1, poss_Rlds[i]])
        n = n+1
    #print(R_Array)
    costs_full = []
    costs_HD_full = []
    costs_HD_p1 = []
    costs_HD_p2 = []
    solutions = []
    for Rlds in R_Array:
        S, E, I, H, R, D, cost_full, costHD_full, costHD_p1, costHD_p2 = itera(params, y0, N,t,Rlds[1:])
        costs_full.append(cost_full)
        costs_HD_full.append(costHD_full)
        costs_HD_p1.append(costHD_p1)
        costs_HD_p2.append(costHD_p2)
        solutions.append([S, E, I, H, R, D, Rlds])
        #if D.min() < 0:
            #print(Rlds)
            #print(D.min())
        #if H.min() <0:
            #print(Rlds)
            #print(H.min())
    #print(costs[1])

    #print(costs)
    minimum_full = 10000000000000000000
    index = 0
    for cost in costs_full:
        if cost < minimum_full:
            min_indexs = index
            minimum_full = cost
        index = index + 1


    min_HD = costs_HD_full[min_indexs]
    min_HD_p1 = costs_HD_p1[min_indexs]
    min_HD_p2 = costs_HD_p2[min_indexs]
    #print(min(costs))
    #print(minimum)
    #print(R_Array[min_indexs])

    minimum_sol = tuple(solutions[int(min_indexs)])
    #print('minimum_sol')
    #print(minimum_sol)
    S, E, I, H, R, D, Rlds = minimum_sol

    Rld = Rlds[:len(t)]


    '''
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E/1000000, 'g', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I/1000000, 'y', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, H/1000000, 'r', alpha=0.5, lw=2, label='In Hospital')
    ax.plot(t, R/1000000, 'c', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, D/1000000, 'k', alpha=0.5, lw=2, label='Dead')
    ax.plot(t, Rld, 'k', alpha=0.5, lw=2, ls='dashed', label='RLD')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (millions)')
    ax.set_ylim(0,5)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', color='w', linewidth=2, linestyle='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.show(block=False)
    #plt.pause(1)
    plt.close()
    '''
    return R_Array[min_indexs], minimum_full, min_HD, min_HD_p1, min_HD_p2, H, D

def run_p3(params, y0, N, t, Rld0, Rld1, Rld2): #calculates the costs when the lcokdown levels for all three periods are specified


    '''
    Rlst = []
    with open('inputs4_0.csv', mode = 'r') as file:

        Rcombos = csv.reader(file)
        next(Rcombos)

        for row in Rcombos:
            Rlst.append(row)

    R_Array = np.array(Rlst).astype(float)
    '''
    poss_Rlds = [0.3,0.4,1.0]

    m = 0
    for i in range(len(poss_Rlds)):
        if poss_Rlds[i] == Rld0:
            m=i
            break
    p = 0
    for i in range(len(poss_Rlds)):
        if poss_Rlds[i] == Rld1:
            p=i
            break

    q = 0
    for i in range(len(poss_Rlds)):
        if poss_Rlds[i] == Rld2:
            q=i
            break
    R_Array = []
    n = 9*m+3*p+q+1

    R_Array = [[n,Rld0,Rld1,Rld2]]

    #print(R_Array)
    costs_full = []
    costs_HD_full = []
    costs_HD_p1 = []
    costs_HD_p2 = []
    solutions = []
    for Rlds in R_Array:
        S, E, I, H, R, D, cost_full, costHD_full, costHD_p1, costHD_p2 = itera(params, y0, N,t,Rlds[1:])
        costs_full.append(cost_full)
        costs_HD_full.append(costHD_full)
        costs_HD_p1.append(costHD_p1)
        costs_HD_p2.append(costHD_p2)
        solutions.append([S, E, I, H, R, D, Rlds])
        #if D.min() < 0:
            #print(Rlds)
            #print(D.min())
        #if H.min() <0:
            #print(Rlds)
            #print(H.min())
    #print(costs[1])

    #print(costs)
    minimum_full = 10000000000000000000
    index = 0
    for cost in costs_full:
        if cost < minimum_full:
            min_indexs = index
            minimum_full = cost
        index = index + 1


    min_HD = costs_HD_full[min_indexs]
    min_HD_p1 = costs_HD_p1[min_indexs]
    min_HD_p2 = costs_HD_p2[min_indexs]
    #print(min(costs))
    #print(minimum)
    #print(R_Array[min_indexs])

    minimum_sol = tuple(solutions[int(min_indexs)])
    #print('minimum_sol')
    #print(minimum_sol)
    S, E, I, H, R, D, Rlds = minimum_sol

    Rld = Rlds[:len(t)]


    '''
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E/1000000, 'g', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I/1000000, 'y', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, H/1000000, 'r', alpha=0.5, lw=2, label='In Hospital')
    ax.plot(t, R/1000000, 'c', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(t, D/1000000, 'k', alpha=0.5, lw=2, label='Dead')
    ax.plot(t, Rld, 'k', alpha=0.5, lw=2, ls='dashed', label='RLD')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (millions)')
    ax.set_ylim(0,5)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(True, which='major', color='w', linewidth=2, linestyle='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    #plt.show(block=False)
    #plt.pause(1)
    plt.close()
    '''
    return R_Array[min_indexs], minimum_full, min_HD, min_HD_p1, min_HD_p2, H, D


def solutions_forced_p1(trials, knowns, dim, z1): #finds the deterministic optimum period 2 lockdown level for each E0,R0, where only the period 1 lockdown level is specified

    E0s,R0s = trials

    H0, D0 = knowns

    poss_Rlds = [0.3,0.4,1.0]

    solutions = np.zeros([dim,dim])
    solutions_p1 = np.zeros([dim,dim])
    solutions_full = []
    costs = np.zeros([dim,dim])
    costs30 = np.zeros([dim,dim])
    sol_listp1 = []
    Hs = []
    Ds = []
    n = 0
    for E0 in tqdm(E0s):
        solutions_full.append([])
        Hs.append([])
        Ds.append([])
        m = 0
        I0 = E0
        for R0 in R0s:
            params, y0, N, t = init_(E0,I0,H0,R0,D0)
            solution, cost_full, cost_HD, cost_HD_p1,cost_HD_p2, H, D = run_p1(params, y0, N, t, z1)
            #print(solution)
            solutions_p1[n,m] = solution[2]
            solutions_full[n].append(solution)
            solutions[n,m] = solution[0]
            costs[n,m] = cost_HD_p1
            Hs[n].append(H[29])
            Ds[n].append(D[29])
            #print(solution[0])
            result = [E0,R0,y0,solution,cost,Hs[n][m],Ds[n][m]]
            #writer.writerow(result)
            m = m+1
            sol_listp1.append(solution[2])
        n = n+1

    #print(sol_list)
    #print(solutions_p1)

    sol_listp1 = set(sol_listp1)
    sol_listp1 = list(sol_listp1)

    #print(sol_listp1)

    breaksE = []
    breaksR = []

    unique_sols = len(sol_listp1)
    for i in range(unique_sols):
        breaksE.append(0)
        breaksR.append(0)


    return solutions_p1, sol_listp1, Hs, Ds, solutions

def solutions_forced_p2(trials, knowns, dim, z1, z2): #finds the determinstic optimum period 3 lockdown level for each E0,R0, where the period 1 and 2 lockdown levels are specified

    E0s,R0s = trials

    H0, D0 = knowns

    poss_Rlds = [0.3,0.4,1.0]

    solutions = np.zeros([dim,dim])
    solutions_p2 = np.zeros([dim,dim])
    solutions_full = []
    costs = np.zeros([dim,dim])
    costs30 = np.zeros([dim,dim])
    sol_listp2 = []
    Hs = []
    Ds = []
    n = 0
    for E0 in tqdm(E0s):
        solutions_full.append([])
        Hs.append([])
        Ds.append([])
        m = 0
        I0 = E0
        for R0 in R0s:
            params, y0, N, t = init_(E0,I0,H0,R0,D0)
            solution, cost_full, cost_HD, cost_HD_p1,cost_HD_p2, H, D = run_p2(params, y0, N, t, z1, z2)
            #print(solution)
            solutions_p2[n,m] = solution[3]
            solutions_full[n].append(solution)
            costs[n,m] = cost_HD_p2
            Hs[n].append(H[59])
            Ds[n].append(D[59])
            #print(solution[0])
            result = [E0,R0,y0,solution,cost,Hs[n][m],Ds[n][m]]
            #writer.writerow(result)
            m = m+1
            sol_listp2.append(solution[3])
        n = n+1

    #print(sol_list)
    #print(solutions_p1)

    sol_listp2 = set(sol_listp2)
    sol_listp2 = list(sol_listp2)

    #print(sol_listp1)

    breaksE = []
    breaksR = []

    unique_sols = len(sol_listp2)
    for i in range(unique_sols):
        breaksE.append(0)
        breaksR.append(0)


    return solutions_p2, sol_listp2, Hs, Ds

def solutions_forced_p3(trials, knowns, dim, z1, z2, z3): #runs run_p3 for each E0 and R0 in period 3 (we don't actually use the solutions presented here, but they have been kept in in case the program is extended)

    E0s,R0s = trials

    H0, D0 = knowns

    poss_Rlds = [0.3,0.4,1.0]

    solutions = np.zeros([dim,dim])
    solutions_p3 = np.zeros([dim,dim])
    solutions_full = []
    costs = np.zeros([dim,dim])
    costs30 = np.zeros([dim,dim])
    sol_listp3 = []
    Hs = []
    Ds = []
    n = 0
    for E0 in tqdm(E0s):
        solutions_full.append([])
        Hs.append([])
        Ds.append([])
        m = 0
        I0 = E0
        for R0 in R0s:
            params, y0, N, t = init_(E0,I0,H0,R0,D0)
            solution, cost_full, cost_HD, cost_HD_p1,cost_HD_p2, H, D = run_p3(params, y0, N, t, z1, z2, z3)
            #print(solution)
            solutions_p3[n,m] = solution[3]
            solutions_full[n].append(solution)
            costs[n,m] = cost_HD
            Hs[n].append(H[89])
            Ds[n].append(D[89])
            #print(solution[0])
            result = [E0,R0,y0,solution,cost,Hs[n][m],Ds[n][m]]
            #writer.writerow(result)
            m = m+1
            sol_listp3.append(solution[3])
        n = n+1

    #print(sol_list)
    #print(solutions_p1)

    sol_listp2 = set(sol_listp3)
    sol_listp2 = list(sol_listp3)

    #print(sol_listp1)

    breaksE = []
    breaksR = []

    unique_sols = len(sol_listp3)
    for i in range(unique_sols):
        breaksE.append(0)
        breaksR.append(0)


    return solutions_p3, sol_listp3, costs,Hs, Ds



def thresholds_p1(dim, probs_matrix, trials, knowns): #defines 'notable' hospital occupancy and death ranges that our system could be in at the end of period 1. Calculates new E0,R0 porbability matrices for each range.
    #collect together the E0s and R0s that result in different optimum next steps
    #instead of a range, this will be a lis/sett of E0.R0

    poss_Rlds = [0.3,0.4,1.0]

    for i in range(len(poss_Rlds)):
        print(cost_ld([poss_Rlds[i]]))


    H_intervals_best_sol = []
    D_intervals_best_sol = []
    new_probs_H = []
    new_probs_D = []
    H_ER_ind = []
    D_ER_ind = []
    H_intervals_cond = []
    D_intervals_cond = []
    total_probs_H = []
    total_probs_D = []
    H_intervals_list = []
    D_intervals_list = []
    total_probs_H_cond = []
    total_probs_D_cond = []
    new_probs_H_cond = []
    new_probs_D_cond = []

    for z in range(len(poss_Rlds)):
        z1 = poss_Rlds[z]
        solutionsp1, sol_listp1, Hs, Ds, solutions = solutions_forced_p1(trials, knowns, dim, z1)
        H_upper = (np.array(Hs)).max()
        #print(H_upper)
        H_lower = (np.array(Hs)).min()
        D_upper = (np.array(Ds)).max()
        D_lower = (np.array(Ds)).min()
        H_intervals = np.geomspace(H_lower,H_upper+0.01,num=dim+1)
        D_intervals = np.geomspace(D_lower,D_upper+0.01,num=dim+1)
        H_intervals_list.append(H_intervals)
        D_intervals_list.append(D_intervals)
        D_primary = []
        probsHp1 = []
        probsDp1 = []
        print(sol_listp1)
        H_intervals_best_sol.append(list(np.linspace(0,0,dim+1)))
        D_intervals_best_sol.append(list(np.linspace(0,0,dim+1)))
        H_intervals_cond.append([])
        D_intervals_cond.append([])
        H_ER_ind.append([])
        D_ER_ind.append([])
        H_ij_ranges= np.zeros([dim,dim])
        for h in range(dim):
            H_ER_ind[z].append([])
            D_ER_ind[z].append([])
        new_probs_H.append(list(np.linspace(0,0,dim)))
        new_probs_D.append(list(np.linspace(0,0,dim)))
        new_probs_H_cond.append([])
        new_probs_D_cond.append([])
        total_probs_H.append([])
        total_probs_D.append([])
        total_probs_H_cond.append([])
        total_probs_D_cond.append([])
        for sol in sol_listp1:
            probsHp1.append(list(np.linspace(0,0,dim)))
            probsDp1.append(list(np.linspace(0,0,dim)))

        for i in range(len(solutionsp1)):
            for j in range(len(solutionsp1)):
                #print(str(solutionsp1[i,j])+'&', end='')
                for s in range(len(sol_listp1)):
                    if solutionsp1[i,j] == sol_listp1[s]:
                        for h in range(len(H_intervals)-1):
                            if (Hs[i][j] >= H_intervals[h]) & (Hs[i][j] < H_intervals[h+1]):
                                #print(str(h)+str(i)+','+str(j))
                                probsHp1[s][h] = probsHp1[s][h] + probs_matrix[i,j]
                                H_ER_ind[z][h].append((i,j))
                                H_ij_ranges[i,j] = h
                                break
                        for d in range(len(D_intervals)-1):
                            if (Ds[i][j] >= D_intervals[d]) & (Ds[i][j] < D_intervals[d+1]):
                                probsDp1[s][d] = probsDp1[s][d] + probs_matrix[i,j]
                                D_ER_ind[z][d].append((i,j))
                                break
                        break
            print('\\')

        print(H_ij_ranges)
        #print(np.array(probsHp1[s][h]).sum())
        print('----')
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                print(str(int(solutions[i,j]))+'&', end='')
            print('\\')

        for h in range(len(H_intervals)-1):
            i_max = 0
            for s in range(len(sol_listp1)):
                if probsHp1[s][h] == max([row[h] for row in probsHp1]):
                    i_max = s
            if probsHp1[i_max][h] == 0:
                H_intervals_best_sol[z][h] = 0.00001
            else:
                H_intervals_best_sol[z][h] = sol_listp1[i_max]
            new_probs_H[z][h] = np.zeros([len(probs_matrix), len(probs_matrix)])
            for tup in H_ER_ind[z][h]:
                i, j = tup
                new_probs_H[z][h][i,j] = probs_matrix[i,j]
            total_probs_H[z].append(new_probs_H[z][h].sum())

            print(H_intervals_best_sol[z][h])

        #create histogram style graph with stacked probability of being in each 'predicted optimum period 2 solution' binned by hospital range.
        print(probsHp1[0])
        fig, (ax0, ax1)= plt.subplots(2, 1, sharex=True,height_ratios=[7, 1])
        labels0 = [str(i) for i in sol_listp1]
        #for s in range(len(sol_listp1)):
        #s    ax.stairs(probsHp1[s],H_intervals, label=(str(sol_listp1[s])))

        norm_factor = np.array(probsHp1).sum()
        print('norm factor')
        print(norm_factor)

        x = []
        x1= []
        x0= []
        H_bars = [h for h in H_intervals]
        for s in range(len(sol_listp1)):
            x1.append([])
            x0.append(H_bars[:-1])
            x.append(H_intervals[:-1])
            for h in range(len(H_intervals)-1):
                if H_intervals_best_sol[z][h] == sol_listp1[s]:
                    x1[s].append(1)
                else:
                    x1[s].append(0)
            size = s*h

        print(x1)
        print(H_intervals)

        print(x)
        '''
        np.random.seed(0)
        data = np.random.normal(50, 20, 10000)
        (counts, bins) = np.histogram(data, bins=range(101))

        factor = 2
        plt.hist(bins[:-1], bins, weights=factor*counts)
        plt.show()
        '''

        ax0.hist(tuple(x), H_intervals,  weights = tuple(probsHp1), histtype='bar', rwidth = 0.95, stacked=True,  label = labels0)

        ax0.set_ylabel('Probability')
        ax0.set_title('')
        ax0.legend()

        ax1.hist(tuple(x), H_intervals, weights = tuple(x1), histtype='bar', rwidth = 0.95, stacked=True)
        ax1.set_yticks([])
        ax1.set_ylabel('Predicted \n best intervention \n for next period ', rotation=0, fontsize = 6)
        ax1.yaxis.set_label_coords(-.08, 0)
        plt.xscale("log")
        plt.xlabel('Hospital Occupancy')
        plt.show()

        index_tracker = []
        for p in range(len(poss_Rlds)):
            index_tracker.append([])
            for h in range(len(H_intervals)-1):
                if H_intervals_best_sol[z][h] == poss_Rlds[p]:
                    index_tracker[p].append(h)


        for p in range(len(index_tracker)):
            H_intervals_cond[z].append([])
            total_probs_H_cond[z].append(0)
            new_probs_H_cond[z].append(np.zeros([dim,dim]))
            for h_ in range(len(index_tracker[p])):
                #print(index_tracker[p][h_])
                #print(H_intervals_list[z])
                H_intervals_cond[z][p].append(H_intervals_list[z][index_tracker[p][h_]])
                total_probs_H_cond[z][p] = total_probs_H_cond[z][p]+total_probs_H[z][index_tracker[p][h_]]
                new_probs_H_cond[z][p] = new_probs_H_cond[z][p]+new_probs_H[z][index_tracker[p][h_]]
        #print(H_intervals_cond[z])



        for d in range(len(D_intervals)-1):
            i_max = 0
            for s in range(len(sol_listp1)):
                if probsDp1[s][d] == max([row[d] for row in probsDp1]):
                    i_max = s
            if probsDp1[i_max][d] == 0:
                D_intervals_best_sol[z][d] = 0.00001
            else:
                D_intervals_best_sol[z][d] = sol_listp1[i_max]
            new_probs_D[z][d] = np.zeros([len(probs_matrix), len(probs_matrix)])
            for tup in D_ER_ind[z][d]:
                i, j = tup
                new_probs_D[z][d][i,j] = probs_matrix[i,j]
            total_probs_D[z].append(new_probs_D[z][d].sum())

        index_tracker = []
        for p in range(len(poss_Rlds)):
            index_tracker.append([])
            for d in range(len(D_intervals)-1):
                if D_intervals_best_sol[z][d] == poss_Rlds[p]:
                    index_tracker[p].append(d)


        for p in range(len(index_tracker)):
            D_intervals_cond[z].append([])
            total_probs_D_cond[z].append(0)
            new_probs_D_cond[z].append(np.zeros([dim,dim]))
            for d_ in range(len(index_tracker[p])):
                D_intervals_cond[z][p].append(D_intervals_list[z][index_tracker[p][d_]])
                total_probs_D_cond[z][p] = total_probs_D_cond[z][p]+total_probs_D[z][index_tracker[p][d_]]
                new_probs_D_cond[z][p] = new_probs_D_cond[z][p]+new_probs_D[z][index_tracker[p][d_]]

    #print(new_probs_H_cond)
    #print(new_probs_H)
    #print(H_intervals)

    #print(probsHp1)
    #print(H_ER_ind)

    #print(new_probs_H)
    #print(total_probs_H)
    #print(sum(total_probs_H[1]))
    #print('new probs')
    #print(new_probs_H[1][2])

    print('done')

    return H_ER_ind, D_ER_ind, H_intervals_cond, D_intervals_cond, total_probs_H_cond, total_probs_D_cond, new_probs_H_cond, new_probs_D_cond

def thresholds_p2(dim, probs_matrix, trials, knowns, H_intervalsp1, D_intervalsp1,new_probs_Hp1, new_probs_Dp1):#defines 'notable' hospital occupancy and death ranges that our system could be in at the end of period 2. Calculates new E0,R0 porbability matrices for each range
    #collect together the E0s and R0s that result in different optimum next steps
    #instead of a range, this will be a lis/sett of E0.R0

    poss_Rlds = [0.3,0.4,1.0]


    H_intervals_cond = []
    D_intervals_cond = []
    total_probs_H_cond = []
    total_probs_D_cond = []
    new_probs_H_cond = []
    new_probs_D_cond = []


    for z in range(len(poss_Rlds)):


        H_intervals_cond.append([])
        D_intervals_cond.append([])
        new_probs_H_cond.append([])
        new_probs_D_cond.append([])
        total_probs_H_cond.append([])
        total_probs_D_cond.append([])

        #print(len(H_intervalsp1[z]))
        #print(len(new_probs_H[z]))
        for hp1 in range(len(H_intervalsp1[z])):#
            #print(hp2)
            probs_matrixp1 = new_probs_Hp1[z][hp1]

            H_intervals_cond[z].append([])
            D_intervals_cond[z].append([])
            new_probs_H_cond[z].append([])
            new_probs_D_cond[z].append([])
            total_probs_H_cond[z].append([])
            total_probs_D_cond[z].append([])
            #print(probs_matrixp1)

            H_intervals_best_sol = []
            D_intervals_best_sol = []
            new_probs_H = []
            new_probs_D = []
            H_ER_ind = []
            D_ER_ind = []
            total_probs_H = []
            total_probs_D = []
            H_intervals_list = []
            D_intervals_list = []

            for zz in range(len(poss_Rlds)):
                z1 = poss_Rlds[z]
                z2 = poss_Rlds[zz]
                solutionsp2, sol_listp2, Hs, Ds = solutions_forced_p2(trials, knowns, dim, z1, z2)
                H_upper = max(max(Hs))
                #print(H_upper)
                H_lower = min(min(Hs))
                D_upper = max(max(Ds))
                D_lower = min(min(Ds))
                H_intervals = np.geomspace(H_lower,H_upper+0.01,num=dim+1)
                D_intervals = np.geomspace(D_lower,D_upper+0.01,num=dim+1)
                H_intervals_list.append(H_intervals)
                D_intervals_list.append(D_intervals)
                D_primary = []
                probsHp2 = []
                probsDp2 = []
                H_intervals_best_sol.append(list(np.linspace(0,0,dim)))
                D_intervals_best_sol.append(list(np.linspace(0,0,dim)))
                H_ER_ind.append([])
                D_ER_ind.append([])
                for h in range(dim):
                    H_ER_ind[zz].append([])
                    D_ER_ind[zz].append([])
                new_probs_H.append(list(np.linspace(0,0,dim)))
                new_probs_D.append(list(np.linspace(0,0,dim)))
                total_probs_H.append([])
                total_probs_D.append([])

                H_intervals_cond[z][hp1].append([])
                D_intervals_cond[z][hp1].append([])
                new_probs_H_cond[z][hp1].append([])
                new_probs_D_cond[z][hp1].append([])
                total_probs_H_cond[z][hp1].append([])
                total_probs_D_cond[z][hp1].append([])

                for sol in sol_listp2:
                    probsHp2.append(list(np.linspace(0,0,dim)))
                    probsDp2.append(list(np.linspace(0,0,dim)))

                for i in range(len(solutionsp2)):
                    for j in range(len(solutionsp2)):
                        for s in range(len(sol_listp2)):
                            if solutionsp2[i,j] == sol_listp2[s]:
                                for h in range(len(H_intervals)-1):
                                    if (Hs[i][j] >= H_intervals[h]) & (Hs[i][j] < H_intervals[h+1]):
                                        probsHp2[s][h] = probsHp2[s][h] + probs_matrixp1[i,j]
                                        H_ER_ind[zz][h].append((i,j))
                                        break
                                for d in range(len(D_intervals)-1):
                                    if (Ds[i][j] >= D_intervals[d]) & (Ds[i][j] < D_intervals[d+1]):
                                        probsDp2[s][d] = probsDp2[s][d] + probs_matrixp1[i,j]
                                        D_ER_ind[zz][d].append((i,j))
                                        break
                                break

                for h in range(len(H_intervals)-1):
                    i_max = 0
                    for s in range(len(sol_listp2)):
                        if probsHp2[s][h] == max([row[h] for row in probsHp2]):
                            i_max = s
                    if probsHp2[i_max][h] == 0:
                        H_intervals_best_sol[zz][h] = 0.00001
                    else:
                        H_intervals_best_sol[zz][h] = sol_listp2[i_max]
                    new_probs_H[zz][h] = np.zeros([len(probs_matrixp1), len(probs_matrixp1)])
                    for tup in H_ER_ind[zz][h]:
                        i, j = tup
                        new_probs_H[zz][h][i,j] = probs_matrixp1[i,j]
                    total_probs_H[zz].append(new_probs_H[zz][h].sum())


                index_tracker = []
                for p in range(len(poss_Rlds)):
                    index_tracker.append([])
                    for h in range(len(H_intervals)-1):
                        if H_intervals_best_sol[zz][h] == poss_Rlds[p]:
                            index_tracker[p].append(h)


                for p in range(len(index_tracker)):
                    H_intervals_cond[z][hp1][zz].append([])
                    total_probs_H_cond[z][hp1][zz].append(0)
                    new_probs_H_cond[z][hp1][zz].append(np.zeros([dim,dim]))
                    for h_ in range(len(index_tracker[p])):
                        #print(index_tracker[p][h_])
                        #print(H_intervals_list[z])
                        H_intervals_cond[z][hp1][zz][p].append(H_intervals_list[zz][index_tracker[p][h_]])
                        total_probs_H_cond[z][hp1][zz][p] = total_probs_H_cond[z][hp1][zz][p]+total_probs_H[zz][index_tracker[p][h_]]
                        new_probs_H_cond[z][hp1][zz][p] = new_probs_H_cond[z][hp1][zz][p]+new_probs_H[zz][index_tracker[p][h_]]
                '''

                #print(H_intervals_cond[z])
                fig, (ax0, ax1)= plt.subplots(2, 1, sharex=True,height_ratios=[7, 1])
                labels0 = [str(i) for i in sol_listp2]
                #for s in range(len(sol_listp1)):
                #s    ax.stairs(probsHp1[s],H_intervals, label=(str(sol_listp1[s])))

                norm_factor = np.array(probsHp2).sum()
                print('norm factor')
                print(norm_factor)


                x = []
                x1= []
                x0= []
                H_bars = [h for h in H_intervals]
                for s in range(len(sol_listp2)):
                    x1.append([])
                    x0.append(H_bars[:-1])
                    x.append(H_intervals[:-1])
                    for h in range(len(H_intervals)-1):
                        if H_intervals_best_sol[zz][h] == sol_listp2[s]:
                            x1[s].append(1)
                        else:
                            x1[s].append(0)
                    size = s*h

                print(x1)
                print(H_intervals)

                print(x)



                ax0.hist(tuple(x), H_intervals,  weights = tuple(probsHp2), histtype='bar', rwidth = 0.95, stacked=True,  label = labels0)

                ax0.set_ylabel('Probability')
                ax0.set_title('')
                ax0.legend()

                ax1.hist(tuple(x), H_intervals, weights = tuple(x1), histtype='bar', rwidth = 0.95, stacked=True)
                ax1.set_yticks([])
                ax1.set_ylabel('Predicted \n best intervention \n for next period ', rotation=0, fontsize = 6)
                ax1.yaxis.set_label_coords(-.08, 0)
                plt.xscale("log")
                plt.xlabel('Hospital Occupancy')
                plt.show()
                '''

                for d in range(len(D_intervals)-1):
                    i_max = 0
                    for s in range(len(sol_listp2)):
                        if probsDp2[s][d] == max([row[d] for row in probsDp2]):
                            i_max = s
                    if probsDp2[i_max][d] == 0:
                        D_intervals_best_sol[zz][d] = 0.00001
                    else:
                        D_intervals_best_sol[zz][d] = sol_listp2[i_max]
                    new_probs_D[zz][d] = np.zeros([len(probs_matrixp1), len(probs_matrixp1)])
                    for tup in D_ER_ind[zz][d]:
                        i, j = tup
                        new_probs_D[zz][d][i,j] = probs_matrixp1[i,j]
                    total_probs_D[zz].append(new_probs_D[zz][d].sum())

                index_tracker = []
                for p in range(len(poss_Rlds)):
                    index_tracker.append([])
                    for d in range(len(D_intervals)-1):
                        if D_intervals_best_sol[zz][d] == poss_Rlds[p]:
                            index_tracker[p].append(d)


                for p in range(len(index_tracker)):
                    D_intervals_cond[z][hp1][zz].append([])
                    total_probs_D_cond[z][hp1][zz].append(0)
                    new_probs_D_cond[z][hp1][zz].append(np.zeros([dim,dim]))
                    for d_ in range(len(index_tracker[p])):
                        #print(index_tracker[p][h_])
                        #print(H_intervals_list[z])
                        D_intervals_cond[z][hp1][zz][p].append(H_intervals_list[zz][index_tracker[p][d_]])
                        total_probs_D_cond[z][hp1][zz][p] = total_probs_D_cond[z][hp1][zz][p]+total_probs_D[zz][index_tracker[p][d_]]
                        new_probs_D_cond[z][hp1][zz][p] = new_probs_D_cond[z][hp1][zz][p]+new_probs_D[zz][index_tracker[p][d_]]

    #print(H_intervals)

    #print(probsHp1)
    #print(H_ER_ind)

    #print(new_probs_H)
    #print(total_probs_H)
    #print(sum(total_probs_H[1][0]))

    print('donep2')

    return H_intervals_cond, D_intervals_cond, total_probs_H_cond, total_probs_D_cond, new_probs_H_cond, new_probs_D_cond


def expected_end_values(trials, knowns, dim, H_intervalsp2, D_intervalsp2, total_probs_Hp2, total_probs_Dp2, new_probs_Hp2, new_probs_Dp2, E0_bounds, R0_bounds): #calculates the expceted end state costs at the end of period 3 for each branch on the SDP

    poss_Rlds = [0.3,0.4,1.0]

    costs_p3 = []

    #calculate expected end values for all starting conditions

    for z in range(len(poss_Rlds)):

        costs_p3.append([])

        for zz in range(len(poss_Rlds)):
            costs_p3[z].append([])

            for zzz in range(len(poss_Rlds)):
                z1 = poss_Rlds[z]
                z2 = poss_Rlds[zz]
                z3 = poss_Rlds[zzz]
                solutionsp3, sol_listp3, costsp3, Hs, Ds = solutions_forced_p3(trials, knowns, dim, z1, z2, z3)
                costs_p3[z][zz].append(costsp3)

    #calculate probabilities for end period 2/start period 3.
    expected_costs = []
    total_probs_Hp2_norm = []
    for z in range(len(poss_Rlds)):
        expected_costs.append([])
        total_probs_Hp2_norm.append([])
        costs_p3.append([])
        for hp1 in range(len(H_intervalsp2[z])):
            expected_costs[z].append([])
            total_probs_Hp2_norm[z].append([])
            for zz in range(len(poss_Rlds)):
                total_probs_Hp2_norm[z][hp1].append([])
                costs_p3[z].append([])
                expected_costs[z][hp1].append([])
                for hp2 in range(len(H_intervalsp2[z][hp1][zz])):
                    expected_costs[z][hp1][zz].append([])
                    norm_factor = sum(total_probs_Hp2[z][hp1][zz])
                    if norm_factor == 0:
                        total_probs_Hp2_norm[z][hp1][zz].append(0)
                    else:
                        total_probs_Hp2_norm[z][hp1][zz].append(total_probs_Hp2[z][hp1][zz][hp2]/norm_factor)

                    for zzz in range(len(poss_Rlds)):
                        norm_factor = total_probs_Hp2[z][hp1][zz][hp2]
                        if norm_factor == 0:
                            cost_ = 10000000000000000000
                        else:
                            cost_ = (costs_p3[z][zz][zzz]*new_probs_Hp2[z][hp1][zz][hp2]/total_probs_Hp2[z][hp1][zz][hp2]).sum()
                        expected_costs[z][hp1][zz][hp2].append(cost_)
                    print('---')
                    print(expected_costs[z][hp1][zz][hp2])
                print(np.array(total_probs_Hp2_norm[z][hp1][zz]).sum())
    print(expected_costs)
    print(total_probs_Hp2_norm)

    #print probabilities, costs, and weighted costs for end of period 3.

    new_probs_example = new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]

    c = plt.imshow(new_probs_example, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Probability matrix')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()


    exp_example = costs_p3[0][0][0]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]
    print('exp_cost1' + str((costs_p3[0][0][0]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]).sum()))

    c = plt.imshow(exp_example, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Weighted Costs Matrix for Period 3: 0.3')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()

    exp_example2 = costs_p3[0][0][1]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]
    print('exp_cost2' + str((costs_p3[0][0][1]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]).sum()))

    c = plt.imshow(exp_example2, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Weighted Costs Matrix for Period 3: 0.4')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()

    exp_example3 = costs_p3[0][0][2]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]
    print('exp_cost3' + str((costs_p3[0][0][2]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]).sum()))

    c = plt.imshow(exp_example3, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Weighted Costs Matrix for Period 3: 1.0')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()

    cost_example = costs_p3[0][0][0]
    print('exp_cost1' + str((costs_p3[0][0][0]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]).sum()))

    c = plt.imshow(cost_example, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Costs Matrix for Period 3: 0.3')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()

    cost_example2 = costs_p3[0][0][1]
    print('exp_cost2' + str((costs_p3[0][0][1]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]).sum()))

    c = plt.imshow(cost_example2, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Costs Matrix for Period 3: 0.4')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()

    cost_example3 = costs_p3[0][0][2]
    print('exp_cost3' + str((costs_p3[0][0][2]*new_probs_Hp2[0][0][0][0]/total_probs_Hp2[0][0][0][0]).sum()))

    c = plt.imshow(cost_example3, extent = [min(E0_bounds), max(E0_bounds), max(R0_bounds), min(R0_bounds)], interpolation ='none')
    plt.colorbar(c)
    plt.title('Costs Matrix for Period 3: 1.0')
    plt.xlabel('E0')
    plt.ylabel('R0')
    plt.show()

    return expected_costs, total_probs_Hp2_norm


def stochastic_program(expected_costs, total_probs_Hp1, total_probs_Hp2_norm): #works backwards through the program, calculating the minimum expected costs, andd therfore the best decision for each decision node



    poss_Rlds = [0.3,0.4,1.0]

    cost_Rlds = []
    best_decisionp1 = []
    best_decision_costsp1 = []
    best_decisionp2 = []
    best_decision_costsp2 = []
    decisions = []
    decision_costs = []

    for Rld in poss_Rlds:
        cost_Rlds.append(cost_ld([Rld]))

    print(cost_Rlds)
    costs_z = []
    best_0 = 10000000000000000000
    index_best_0 = 0
    exp_costs_hp0 =[]

    for z in range(len(poss_Rlds)):
        best_decisionp2.append([])
        best_decisionp1.append([])
        best_decision_costsp1.append([])
        best_decision_costsp2.append([])
        exp_costs_hp1 = []
        for hp1 in range(len(total_probs_Hp1)):
            best_decisionp2[z].append([])
            best_decision_costsp2[z].append([])

            costs_zz = []
            best_1 = 10000000000000000000
            index_best_1 = 0


            for zz in range(len(poss_Rlds)):
                best_decisionp2[z][hp1].append([])
                best_decision_costsp2[z][hp1].append([])
                exp_costs_hp2 = []
                for hp2 in range(len(total_probs_Hp2_norm[z])):
                    best_2 = 10000000000000000000
                    index_best_2 = 0

                    for zzz in range(len(expected_costs[z][hp1][zz][hp2])):
                        test_p2 = expected_costs[z][hp1][zz][hp2][zzz]+cost_Rlds[zzz]
                        if (test_p2 < best_2) & (test_p2 > 0):
                            best_2 = test_p2
                            index_best_2 = zzz
                    best_decisionp2[z][hp1][zz].append(poss_Rlds[index_best_2])
                    #print(best_decisionp2)
                    best_decision_costsp2[z][hp1][zz].append(best_2)
                    expected_costp2 = total_probs_Hp2_norm[z][hp1][zz][hp2]*best_2
                    exp_costs_hp2.append(expected_costp2)
                costs_zz.append(sum(exp_costs_hp2))
                test_p1 = costs_zz[zz]+cost_Rlds[zz]
                print(costs_zz[zz])
                print(test_p1)
                if (test_p1 < best_1) & (test_p1 > 0) :
                    best_1 = test_p1
                    index_best_1 = zz
            best_decisionp1[z].append(poss_Rlds[index_best_1])
            best_decision_costsp1[z].append(best_1)
            expected_costsp1 = total_probs_Hp1[z][hp1]*best_1
            exp_costs_hp1.append(expected_costsp1)
        costs_z.append(sum(exp_costs_hp1))
        test_p0 = costs_z[z] + cost_Rlds[z]
        if (test_p0 < best_0) & (test_p0 > 0):
            best_0 = costs_z[z]+cost_Rlds[z]
            index_best_0 = z
    best_decisionp0 = poss_Rlds[index_best_0]
    best_decision_costsp0 = best_0


    f = open("results_tree_size_15_.txt","w+")
    f.write("results \n decision period 2: \n")

    f.write("expected cost of best decision"+str(best_decision_costsp2)+"\n")
    f.write("best decision for each hos range"+str(best_decisionp2)+"\n")
    f.write("Hos range probs"+str(total_probs_Hp2_norm)+"\n")
    f.write("decision period 1: \n")

    f.write("expected cost of best decision"+str(best_decision_costsp1)+"\n")
    f.write("best decision for each hos range"+str(best_decisionp1)+"\n")
    f.write("Hos range probs"+str(total_probs_Hp1)+"\n")
    f.write("decision period 0: \n")

    f.write("expected cost of best decision"+str(best_decision_costsp0)+"\n")
    f.write("best intial decision"+str(best_decisionp0)+"\n")


    f.close()


def main(start): #runs program


    dim = 15
    H0 = 200
    D0 = 100
    E0s = np.linspace(0,300000,dim)
    R0s = np.linspace(0,300000,dim)

    distE = (100000,300000)
    distR = (250000,300000)


    trials = (E0s, R0s)
    knowns = (H0,D0)

    E0_bounds, R0_bounds = boundaries(E0s,R0s)

    E_probs, R_probs, probs_matrix = probabilities_func(E0_bounds,R0_bounds,distE,distR,E0s,R0s)


    H_ER_ind, D_ER_ind, H_intervalsp1, D_intervalsp1, total_probs_Hp1, total_probs_Dp1, new_probs_Hp1, new_probs_Dp1 = thresholds_p1(dim, probs_matrix, trials, knowns)
    H_intervalsp2, D_intervalsp2, total_probs_Hp2, total_probs_Dp2, new_probs_Hp2, new_probs_Dp2 = thresholds_p2(dim, probs_matrix, trials, knowns, H_intervalsp1, D_intervalsp1,new_probs_Hp1, new_probs_Dp1)

    expected_costs, total_probs_Hp2_norm = expected_end_values(trials, knowns, dim, H_intervalsp2, D_intervalsp2, total_probs_Hp2, total_probs_Dp2, new_probs_Hp2, new_probs_Dp2, E0_bounds, R0_bounds)


    end = time.time()

    elapsed_time = end - start

    print('run time')
    print(elapsed_time)

    f = open("expected_costs2.txt", "w+")
    f.write(str(expected_costs))
    f.close()

    g = open("total_probs_Hp2_norm.txt","w+")
    g.write(str(total_probs_Hp2_norm))
    g.close()

    m = open("total_probs_Hp1.txt","w+")
    m.write(str(total_probs_Hp1))
    m.close()

    n = open("H_intervalsp1.txt","w+")
    n.write(str(H_intervalsp1))
    n.close()

    o = open("H_intervalsp2.txt","w+")
    o.write(str(H_intervalsp2))
    o.close()

    stochastic_program(expected_costs, total_probs_Hp1, total_probs_Hp2_norm)






main(start)
