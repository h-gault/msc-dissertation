import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
import csv

def deriv(y, t, paras): #intialize and compute result of differential equations

    #S = y[0]
    E = y[0]
    I = y[1]
    H = y[2]
    R = y[3]
    D_c = y[4]
    D_h = y[5]

    Rld = 1


    try:
        N = paras['N'].value
        i = paras['i'].value
        h = paras['h'].value
        x = paras['x'].value
        r = paras['r'].value
        z = paras['z'].value
        p = paras['p'].value
        d = paras['d'].value
        f = paras['f'].value
        k = paras['k'].value
        R_0 = paras['R_0'].value

    except KeyError:
        N,i, h, x, r, z, p, d, f, k, R_0 = paras

    S = N - E - I - H -R - D_c - D_h

    e =R_0*Rld*(d+h+r)*100000/10000
    # the model equations
    dSdt = k*R -e*S*I/N
    dEdt = -i*E-x*E+e*I*S/N
    dIdt = i*E-d*I-h*I+p*H-r*I
    dHdt = h*I-p*H-f*H-z*H
    dRdt = z*H+r*I-k*R+x*E
    dDcdt = d*I
    dDhdt = f*H


    return_vals = [dEdt, dIdt, dHdt, dRdt, dDcdt, dDhdt]



    return return_vals


def g(t, x0, paras): #soltution to the ODE for the given parameters

    ret = odeint(deriv, x0, t, args=(paras,))
    E, I, H, R, D_c, D_h = ret.T

    N = paras['N'].value
    S = N - E - I - H -R - D_c - D_h
    return E, I, H, R, D_c, D_h


def residual(paras, t, H_data): #compute the residual between the actual and fitted data


    x0 = paras['E0'].value, paras['I0'].value, paras['H0'].value, paras['R0'].value, paras['D_c0'].value, paras['D_h0'].value
    E, I, H, R, D_c, D_h = g(t, x0, paras)

    # only use data for H for the git
    H_model = H


    return (H_model - H_data).ravel()



def main_fits():

    # initial conditions
    N = 5000000
    delayplus = 0
    delayneg = 31



    # measured data

    data_list= []
    with open('datainput1.csv', mode = 'r') as file:

        data = csv.reader(file)
        next(data)

        for row in data:

            data_list.append(row[delayneg+1:])

    # Initial number of infected and recovered individuals, I0 and R0.

    E0, I0, H0, R0, D_c0, D_h0 = 10000, 10000, float(data_list[1][delayneg]), 1000, sum([float(i) for i in data_list[3][:delayneg]]), sum([float(i) for i in data_list[5][:delayneg]])
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0 - E0 - float(H0) - float(D_c0) - float(D_h0)

    y0 = [E0, I0, float(H0), R0, float(D_c0), float(D_h0)]
    print(y0)


    t_measured = np.linspace(0, len(data_list[1][:50])+delayplus, len(data_list[1][:50])+delayplus)
    delay_arr = np.zeros([delayplus,1])
    print(delay_arr)
    H_measured = np.append(delay_arr,np.array(data_list[1][:50]).astype(float))


    plt.figure()
    plt.scatter(t_measured, H_measured, marker='o', color='r', label='measured H data', s=75)



    # set parameters
    params = Parameters()
    #params.add('S0', value=S0, vary = False)
    params.add('E0', value=E0, min =0, max = 1000000, vary = True)
    params.add('I0', value=I0, min = 0, max = 1000000, vary = True)
    params.add('H0', value=float(H0),min = 0.9*float(H0),max = 1.1*float(H0)+0.01, vary=True)
    params.add('R0', value=R0, min = 0,max =1000000, vary=True)
    params.add('D_c0', value=float(D_c0),min = 0.9*float(D_c0),max = 1.1*float(D_c0)+0.01, vary=True)
    params.add('D_h0', value=float(D_h0),min = 0.9*float(D_h0),max = 1.1*float(D_h0)+0.01, vary=True)
    params.add('N', value=N, vary=False)
    params.add('i', value=3, min = 0.1, max = 10)
    params.add('h', value=0.01, min=0.0001, max = 10)
    params.add('x', value=0, vary=False)
    params.add('r', value=0.7, min = 0.1, max = 1, vary=True)
    params.add('z', value=0.1, min=0.0001, max=1, vary = True)
    params.add('p', value=0, vary=False)
    params.add('d', value=0.001, min = 0.0001, max = 1, vary=True)
    params.add('f', value=0.1, min= 0.001, max = 1, vary=False)
    params.add('k', value=0.01, min = 0.0001, max = 1, vary = True)
    params.add('R_0', value=2.5, min = 1, max = 10, vary=True)

    #fit model using scipy.minimize
    result = minimize(residual, params, args=(t_measured, H_measured), method='leastsq')  # leastsq nelder
    print('result')
    print(result.params)
    list_ans = list(result.params.values())
    parameters_ = []
    for i in range(8,len(list_ans)):
        parameters_.append(list_ans[i].value)
    print(parameters_)


    # check results of the fit
    E_fitted, I_fitted, H_fitted, R_fitted, D_cfitted, D_hfitted = g(np.linspace(0, 100+delayplus-delayneg, 101+delayplus-delayneg), y0, result.params)

    S_fitted = N - E_fitted - I_fitted - H_fitted -  R_fitted - D_cfitted - D_hfitted
    print(H_fitted)
    # plot the fitted data

    plt.plot(np.linspace(0, 100+delayplus-delayneg, 101+delayplus-delayneg), H_fitted, '-', linewidth=2, color='red', label='fitted H data')

    plt.legend()
    plt.xlim([0, max(t_measured)])
    plt.ylim([0, max(H_fitted)])


    report_fit(result)
    plt.xlabel('Time (days)')
    plt.ylabel('Hospital occupancy')
    plt.show()

    return tuple(parameters_)

main_fits()
