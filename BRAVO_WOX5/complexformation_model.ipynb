import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import matplotlib
from matplotlib import rc
import pandas as pd 
import seaborn as sns

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['mathtext.fontset'] = 'cm'

start = time.time()

fig = plt.figure(1)

A0 = 10; A = 0.5; DM = 0.01; DC = 0.001; KON = 0.01; KOFF = 0.02; EM = 0.5; EW = 10;
G0 = 10; G = 5; DW = 0.01; W0 = 83*5;
KW = 0.005; KM = 0.1; n = 2; m = 2;

AT = 1; DT = 0.01;
KONBT = 0.01; KOFFBT = 0.01; DCBT = 0.001;
KONWT = 0.0; KOFFWT = 0.01; DCWT = 0.001;

N = 1000

sm_exp = 1.85834610;
smd_exp = 0.535182514;
smdd_exp = 0.614542045;
sw_exp = 1.311341103;
swd_exp = 0.822331879;
swdd_exp = 1.345341568;
swoe_exp = 1;
smoe_exp = 2.3;


l = np.logspace(-10,10,N)


def equations_wt(X):
    fwt = [a*((1 + em*(km*X[0])**n )/(1 + (km*X[0])**n ))*((1 + ew*(kw*X[1])**n )/(1 + (kw*X[1])**n )) - kon*X[0]*X[1] + koff*X[2] - konbt*X[0]*X[3] - koffbt*X[4] - dm*X[0]]
    fwt.append(g*(w0**m)/(w0**m + (X[1])**m ) - kon*X[0]*X[1] + koff*X[2] - konwt*X[1]*X[3] + koffwt*X[5] - dw*X[1])
    fwt.append(kon*X[0]*X[1] - koff*X[2]  - dc*X[2])
    fwt.append(at - dt*X[3] - konbt*X[0]*X[3] + koffbt*X[4] - konwt*X[1]*X[3] + koffwt*X[5])
    fwt.append(konbt*X[0]*X[3] - koffbt*X[4] - dcbt*X[4])
    fwt.append(konwt*X[1]*X[3] - koffwt*X[5] - dcwt*X[5])

    return fwt

def equations_bm(X):
    fbm = [X[0]]
    fbm.append(g*(w0**m)/(w0**m + (X[1])**m) - konwt*X[1]*X[3] + koffwt*X[5] - dw*X[1])
    fbm.append(X[2])
    fbm.append(at - dt*X[3] - konwt*X[1]*X[3] + koffwt*X[5])
    fbm.append(X[4])
    fbm.append(konwt*X[1]*X[3] - koffwt*X[5] - dcwt*X[5])

    return fbm

def equations_wm(X):
    fwm = [a*((1 + em*(km*X[0])**2)/(1 + (km*X[0])**2)) - konbt*X[0]*X[3] - koffbt*X[4] - dm*X[0]]
    fwm.append(X[1])
    fwm.append(X[2])
    fwm.append(at - dt*X[3] - konbt*X[0]*X[3] + koffbt*X[4])
    fwm.append(konbt*X[0]*X[3] - koffbt*X[4] - dcbt*X[4])
    fwm.append(X[5])
    return fwm

def equations_dm(X):
    fdm = [X[0]]
    fdm.append(X[1])
    fdm.append(X[2])
    fdm.append(at - dt*X[3])
    fdm.append(X[4])
    fdm.append(X[5])
    return fdm

def equations_boe(X):
    fboe = [a0*a + a*((1 + em*(km*X[0])**n )/(1 + (km*X[0])**n ))*((1 + ew*(kw*X[1])**n )/(1 + (kw*X[1])**n )) - kon*X[0]*X[1] + koff*X[2] - konbt*X[0]*X[3] - koffbt*X[4] - dm*X[0]]
    fboe.append(g*(w0**m)/(w0**m + (X[1])**m ) - kon*X[0]*X[1] + koff*X[2] - konwt*X[1]*X[3] + koffwt*X[5] - dw*X[1])
    fboe.append(kon*X[0]*X[1] - koff*X[2]  - dc*X[2])
    fboe.append(at - dt*X[3] - konbt*X[0]*X[3] + koffbt*X[4] - konwt*X[1]*X[3] + koffwt*X[5])
    fboe.append(konbt*X[0]*X[3] - koffbt*X[4] - dcbt*X[4])
    fboe.append(konwt*X[1]*X[3] - koffwt*X[5] - dcwt*X[5])
    return fboe

def equations_woe(X):
    fwoe = [a*((1 + em*(km*X[0])**n )/(1 + (km*X[0])**n ))*((1 + ew*(kw*X[1])**n )/(1 + (kw*X[1])**n )) - kon*X[0]*X[1] + koff*X[2] - konbt*X[0]*X[3] - koffbt*X[4] - dm*X[0]]
    fwoe.append(g0*g + g*(w0**m)/(w0**m + (X[1])**m ) - kon*X[0]*X[1] + koff*X[2] - konwt*X[1]*X[3] + koffwt*X[5] - dw*X[1])
    fwoe.append(kon*X[0]*X[1] - koff*X[2]  - dc*X[2])
    fwoe.append(at - dt*X[3] - konbt*X[0]*X[3] + koffbt*X[4] - konwt*X[1]*X[3] + koffwt*X[5])
    fwoe.append(konbt*X[0]*X[3] - koffbt*X[4] - dcbt*X[4])
    fwoe.append(konwt*X[1]*X[3] - koffwt*X[5] - dcwt*X[5])
    return fwoe


factor = 2;


Xwt = np.zeros(N)
Zwt = np.zeros(N)
Twt = np.zeros(N)
CBWwt = np.zeros(N)
CBTwt = np.zeros(N)
CWTwt = np.zeros(N)

Xbm = np.zeros(N)
Zbm = np.zeros(N)
Tbm = np.zeros(N)
CBWbm = np.zeros(N)
CBTbm = np.zeros(N)
CWTbm = np.zeros(N)

Xwm = np.zeros(N)
Zwm = np.zeros(N)
Twm = np.zeros(N)
CBWwm = np.zeros(N)
CBTwm = np.zeros(N)
CWTwm = np.zeros(N)

Xdm = np.zeros(N)
Zdm = np.zeros(N)
Tdm = np.zeros(N)
CBWdm = np.zeros(N)
CBTdm = np.zeros(N)
CWTdm = np.zeros(N)

Xboe = np.zeros(N)
Zboe = np.zeros(N)
Tboe = np.zeros(N)
CBWboe = np.zeros(N)
CBTboe = np.zeros(N)
CWTboe = np.zeros(N)

Xwoe = np.zeros(N)
Zwoe = np.zeros(N)
Twoe = np.zeros(N)
CBWwoe = np.zeros(N)
CBTwoe = np.zeros(N)
CWTwoe = np.zeros(N)

pm = np.zeros(N)
pw = np.zeros(N)
sm = np.zeros(N)
sw = np.zeros(N)
smd = np.zeros(N)
swd = np.zeros(N)
smdd = np.zeros(N)
swdd = np.zeros(N)

swoe = np.zeros(N)
smoe = np.zeros(N)

result_wt = [1,1000,1,100,1,1]
result_bm = [0,1000,0,100,0,1]
result_wm = [100,0,0,100,1,0]
result_dm = [0,0,0,100,0,0]
result_boe = [1000,1,100,1,1,1]
result_woe = [1,1000,1,100,1,1]


for i in range(0,N):

    ew = np.random.uniform(EW/factor,factor*EW)
    em = np.random.uniform(EM/factor,factor*EM)
    km = np.random.uniform(KM/factor,factor*KM)
    kw = np.random.uniform(KW/factor,factor*KW)
    w0 = np.random.uniform(W0/factor,factor*W0)
    a = np.random.uniform(A/factor,factor*A)
    g = np.random.uniform(G/factor,factor*G)
    dw = DW + 0*np.random.uniform(DW/factor,factor*DW)
    dm = DM + 0*np.random.uniform(DM/factor,factor*DM)
    a0 = np.random.uniform(A0/factor,factor*A0)
    g0 = np.random.uniform(G0/factor,factor*G0)
    kon = np.random.uniform(KON/factor,factor*KON)
    koff = KOFF + 0*np.random.uniform(KOFF/factor,factor*KOFF)
    dc = DC + 0*np.random.uniform(DC/factor,factor*DC)
    
    at = np.random.uniform(AT/factor,factor*AT)
    dt = DT + 0*np.random.uniform(DT/factor,factor*DT)
    konwt = np.random.uniform(KONWT/factor,factor*KONWT)
    konbt = np.random.uniform(KONBT/factor,factor*KONBT)
    koffwt = KOFFWT + 0*np.random.uniform(KOFFWT/factor,factor*KOFFWT)
    koffbt = KOFFBT + 0*np.random.uniform(KOFFBT/factor,factor*KOFFBT)
    dcbt = DCBT + 0*np.random.uniform(DCBT/factor,factor*DCBT)
    dcwt = DCWT + 0*np.random.uniform(DCWT/factor,factor*DCWT)


    result_wt = fsolve(equations_wt, result_wt)
    Xwt[i] = result_wt[0]     ###BRAVO steady state
    Zwt[i] = result_wt[1]     ###WOX5 steady state
    CBWwt[i] = result_wt[2]   ###CBW steady state
    Twt[i] = result_wt[3]     ###TPL steady state
    CBTwt[i] = result_wt[4]   ###CBT steady state
    CWTwt[i] = result_wt[5]   ###CWT steady state

    result_bm = fsolve(equations_bm, result_bm)
    Xbm[i] = result_bm[0]   ###BRAVO steady state
    Zbm[i] = result_bm[1]    ###WOX5 steady state
    CBWbm[i] = result_bm[2]   ###CBW steady state
    Tbm[i] = result_bm[3]     ###TPL steady state
    CBTbm[i] = result_bm[4]   ###CBT steady state
    CWTbm[i] = result_bm[5]   ###CWT steady state

    result_wm = fsolve(equations_wm, result_wm)
    Xwm[i] = result_wm[0]   ###BRAVO steady state
    Zwm[i] = result_wm[1]    ###WOX5 steady state
    CBWwm[i] = result_wm[2]   ###CBW steady state
    Twm[i] = result_wm[3]     ###TPL steady state
    CBTwm[i] = result_wm[4]   ###CBT steady state
    CWTwm[i] = result_wm[5]   ###CWT steady state

    result_dm = fsolve(equations_dm, result_dm)
    Xdm[i] = result_dm[0]   ###BRAVO steady state
    Zdm[i] = result_dm[1]    ###WOX5 steady state
    CBWdm[i] = result_dm[2]   ###CBW steady state
    Tdm[i] = result_dm[3]     ###TPL steady state
    CBTdm[i] = result_dm[4]   ###CBT steady state
    CWTdm[i] = result_dm[5]   ###CWT steady state
    
    result_boe = fsolve(equations_boe, result_boe)
    Xboe[i] = result_boe[0]   ###BRAVO steady state
    Zboe[i] = result_boe[1]    ###WOX5 steady state
    CBWboe[i] = result_boe[2]   ###CBW steady state
    Tboe[i] = result_boe[3]     ###TPL steady state
    CBTboe[i] = result_boe[4]   ###CBT steady state
    CWTboe[i] = result_boe[5]   ###CWT steady state
    
    result_woe = fsolve(equations_woe, result_woe)
    Xwoe[i] = result_woe[0]   ###BRAVO steady state
    Zwoe[i] = result_woe[1]    ###WOX5 steady state
    CBWwoe[i] = result_woe[2]   ###CBW steady state
    Twoe[i] = result_woe[3]     ###TPL steady state
    CBTwoe[i] = result_woe[4]   ###CBT steady state
    CWTwoe[i] = result_woe[5]   ###CWT steady state
    
    pm[i] = a*((1 + em*(km*Xwt[i] )**n)/(1 + (km*Xwt[i] )**n))*((1 + ew*(kw*Zwt[i] )**n )/(1 + (kw*Zwt[i] )**n))
    pw[i]  = g*(w0**m)/(w0**m + (Zwt[i] )**m)

    sm[i]  = (a*(1 + ew*(kw*Zbm[i] )**n)/(1 + (kw*Zbm[i] )**n))/pm[i]
    sw[i]  = g/pw[i]
    
    smd[i]  = (a*(1 + em*(km*Xwm[i] )**n)/(1 + (km*Xwm[i] )**n))/pm[i]
    swd[i]  = (g*(w0**m)/(w0**m + (Zbm[i] )**m))/pw[i]
    
    smdd[i]  = a/pm[i]
    swdd[i]  = g/pw[i]
    
    swoe[i] = g*(w0**m)/(w0**m + (Zboe[i] )**m)/pw[i]
    smoe[i] = (a*((1 + em*(km*Xwoe[i] )**n)/(1 + (km*Xwoe[i] )**n))*((1 + ew*(kw*Zwoe[i] )**n )/(1 + (kw*Zwoe[i] )**n)))/pm[i]

FCs = pd.DataFrame()
FCs['$\sigma_B$'] = pd.Series(sm)
FCs['$\sigma_B^{\dagger}$'] = pd.Series(smd)
FCs['$\sigma_B^{\dagger\dagger}$'] = pd.Series(smdd)
FCs['$\sigma_B^{Woe}$'] = pd.Series(smoe)
FCs['$\sigma_W^{\dagger}$'] = pd.Series(swd)
FCs['$\sigma_W$'] = pd.Series(sw)
FCs['$\sigma_W^{\dagger\dagger}$'] = pd.Series(swdd)
FCs['$\sigma_W^{Boe}$'] = pd.Series(swoe)


line = np.linspace(-10,10,100)
line1 = 0*line + 1

#ax = sns.violinplot(data = FCs, scale = 'width', palette = ['C3','C3','C3','C3','C0','C0','C0','C0'], 
#                    linewidth=1.5, inner = 'quartile')

ax = sns.boxplot(data = FCs, palette = ['C3','C3','C3','C3','C0','C0','C0','C0'], fliersize = 0)
ax = sns.stripplot(data = FCs, jitter = 0.4, size = 3, palette = ['C3','C3','C3','C3','C0','C0','C0','C0'],
                   linewidth=0.3, edgecolor = 'k', alpha = 0.1)

ax.plot(line,line1,'k-', alpha = 0.5)

#ax.set_yscale('log')
ax.set_xlim(-0.5,7.5)
ax.set_ylim(0,6)
ax.grid(alpha = 0.2, linewidth = 2)
ax.set_ylabel('promoter fold-change')

#if at == 0:
#    ax.annotate('complex \nformation', xy = (5,4.5), fontsize = 15)
#else:
#    ax.annotate('complex formation\nwith TOPLESS', xy = (4,4.5), fontsize = 15)

#ax.annotate(r'$a_T = %0.1f$' %AT, xy = (0.1,100), fontsize = 18)

#ax.errorbar(0, sm_exp, yerr=0.642183745, fmt='sC3', mec='k', ecolor = 'k', ms = 7, capsize=4, capthick = 1.5, elinewidth=1.5)
#ax.errorbar(1, smd_exp, yerr=0.213965032, fmt='sC3', mec='k', ecolor = 'k', ms = 7, capsize=4, capthick = 1.5, elinewidth=1.5)
#ax.errorbar(2, smdd_exp, yerr=0.303778688, fmt='sC3', mec='k', ecolor = 'k', ms = 7,  capsize=4, capthick = 1.5, elinewidth=1.5)
#ax.errorbar(3, smoe_exp, yerr=0, fmt='sC3', mec='k', ecolor = 'k', ms = 7,  capsize=4, capthick = 1.5, elinewidth=1.5)
#ax.errorbar(4, sw_exp, yerr=0.171382472, fmt='sC0', mec='k', ecolor = 'k', ms = 7,  capsize=4, capthick = 1.5, elinewidth=1.5)
#ax.errorbar(5, swd_exp, yerr=0.174257429, fmt='sC0', mec='k', ecolor = 'k', ms = 7,  capsize=4, capthick = 1.5, elinewidth=1.5)
#ax.errorbar(6, swdd_exp, yerr=0.253786136, fmt='sC0', mec='k', ecolor = 'k', ms = 7,  capsize=4, capthick = 1.5, elinewidth=1.5)
#ax.errorbar(7, swoe_exp, yerr=0, fmt='sC0', mec='k', ecolor = 'k', ms = 7,  capsize=4, capthick = 1.5, elinewidth=1.5)

plt.savefig('sensitivity_analysis_2_complex.pdf', transparent = True, bbox_inches='tight')

plt.show()

end = time.time()

print(end - start)
