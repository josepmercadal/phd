import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
import matplotlib
from matplotlib import rc

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['mathtext.fontset'] = 'cm'

start = time.time()

fig = plt.figure(1)

a0 = 0.; ##background
A = 0.3; G = 5;
a = 0.3; dm = 0.01; dc = 0.001; kon = 0.0; koff = 0; em = 0.; ew = 12;
g0 = 0.0; g = 5; dw = 0.01; m0 = 10; w0 = 800; w1 = 1;
kw = 0.0015; km = 0.03; n = 2; m = 2;

N = 500

l = np.logspace(-10,10,N)

f, ((ax1, ax2, ax3, ax4), (ax6, ax7, ax8, ax9), (ax10, ax11, ax12, ax13), (ax14, ax15, ax16, ax17)) = plt.subplots(4,4, sharex=True, sharey='row', figsize = (13,13))
f.subplots_adjust(wspace = 0.1, hspace=0.18)

#ax1.axvspan(3, 10, alpha=0.2, color='green')

def equations_wt(X):
    fwt = [a*((1 + em*(km*X[0])**n )/(1 + (km*X[0])**n ))*((1 + ew*(kw*X[1])**n )/(1 + (kw*X[1])**n )) - kon*X[0]*X[1] + koff*X[2] - dm*X[0]]
    fwt.append(g0 + g*(w0**m)/(w0**m + (X[1]*((m0**m)/(m0**m + X[0]**m) + w1))**m ) - kon*X[0]*X[1] + koff*X[2] - dw*X[1])
    fwt.append(kon*X[0]*X[1] - koff*X[2]  - dc*X[2])
    return fwt

def equations_bm(X):
    fbm = [X[0]]
    fbm.append(g0 + g*(w0**m)/(w0**m + (X[1]*((m0**m)/m0**m + w1))**m) - dw*X[1])
    fbm.append(X[2])
    return fbm

def equations_wm(X):
    fwm = [a*((1 + em*(km*X[0])**2)/(1 + (km*X[0])**2)) - dm*X[0]]
    fwm.append(X[1])
    fwm.append(X[2])
    return fwm

def equations_dm(X):
    fdm = [X[0]]
    fdm.append(X[1])
    fdm.append(X[2])
    return fdm

L = np.zeros((N,6))

aa = np.zeros((N,6))
gg = np.zeros((N,6))


Xwt = np.zeros((N,6))
Zwt = np.zeros((N,6))

Xbm = np.zeros((N,6))
Zbm = np.zeros((N,6))

Xwm = np.zeros((N,6))
Zwm = np.zeros((N,6))

Xdm = np.zeros((N,6))
Zdm = np.zeros((N,6))

j = 0

KK = [0.15,0.5,0.7,0.9]

for jj in KK:
    
    j = j + 1

    em = jj
    
    result_wt = [1e6,1,0]
    result_bm = [0,1,0]
    result_wm = [1e6,0,0]
    result_dm = [0,0,0]
        
    for i in range(1,N):
        
        x = 2**(0.1*i - 20)
        a = A/x
        g = 10*G*x/(x+9)
        L[i,j] = x
        
        aa[i,j] = a 
        gg[i,j] = g
                
        #print(a,g,x)
                                        
        result_wt = fsolve(equations_wt, result_wt)
        Xwt[i,j] = result_wt[0]   ###BRAVO steady state
        Zwt[i,j] = result_wt[1]    ###WOX5 steady state
        
        result_bm = fsolve(equations_bm, result_bm)
        Xbm[i,j] = result_bm[0]   ###BRAVO steady state
        Zbm[i,j] = result_bm[1]    ###WOX5 steady state
                
        result_wm = fsolve(equations_wm, result_wm)
        Xwm[i,j] = result_wm[0]   ###BRAVO steady state
        Zwm[i,j] = result_wm[1]    ###WOX5 steady state
        
        result_dm = fsolve(equations_dm, result_dm)
        Xdm[i,j] = result_dm[0]   ###BRAVO steady state
        Zdm[i,j] = result_dm[1]    ###WOX5 steady state
        
        #print(x,Xwt[i,j])
                                    
    ax = globals()["ax" + str(j)]
    axb = globals()["ax" + str(j + 5)]
    axc = globals()["ax" + str(j + 9)]
    axd = globals()["ax" + str(j + 13)]

    
    pm = a0 + aa[:,j]*((1 + em*(km*Xwt[:,j])**n)/(1 + (km*Xwt[:,j])**n))*((1 + ew*(kw*Zwt[:,j])**n )/(1 + (kw*Zwt[:,j])**n))
    pw = g0 + gg[:,j]*(w0**m)/(w0**m + (Zwt[:,j]*((m0**m)/(m0**m + Xwt[:,j]**m) + w1))**m)

    sm = (a0 + aa[:,j]*(1 + ew*(kw*Zbm[:,j])**n)/(1 + (kw*Zbm[:,j])**n))/pm
    sw = (g0 + gg[:,j]*(w0**m)/w0**m)/pw
    
    smd = (a0 + aa[:,j]*(1 + em*(km*Xwm[:,j])**n)/(1 + (km*Xwm[:,j])**n))/pm
    swd = (g0 + gg[:,j]*(w0**m)/(w0**m + (Zbm[:,j]*((m0**m)/m0**m + w1))**m))/pw
    
    smdd = (a0 + aa[:,j])/pm
    swdd = (g0 + gg[:,j]*(w0**m)/w0**m)/pw
           
    line = 1 + l*0
    
    ax.plot(L[:,j],sm,'C3-.', alpha = 1, linewidth = 3, label = r'$\sigma_B$')
    ax.plot(L[:,j],sw,'C0', linestyle = '-.', alpha = 1, linewidth = 3, label = r'$\sigma_W$')
    ax.plot(L[:,j],smd,'C3--', alpha = 1, linewidth = 3, label = r'$\sigma_B^{\dagger}$')
    ax.plot(L[:,j],swd,'C0', linestyle = '--', alpha = 1, linewidth = 3, label = r'$\sigma_W^{\dagger}$')
    ax.plot(L[:,j],smdd,'C3:', alpha = 1, linewidth = 3, label = r'$\sigma_B^{\dagger\dagger}$')
    ax.plot(L[:,j],swdd,'C0', linestyle = ':', alpha = 1, linewidth = 3, label = r'$\sigma_W^{\dagger\dagger}$')
    axb.plot(L[:,j],Xwm[:,j]/Xwt[:,j],'k', alpha = 1, linewidth = 3, label = r'$FC_B$')
    axb.plot(L[:,j],Zbm[:,j]/Zwt[:,j],'k--', alpha = 1, linewidth = 3, label = r'$FC_W$')
    axc.plot(L[:,j],pm,'C3', alpha = 1, linewidth = 3, label = r'$P_B$')
    axc.plot(L[:,j],pw,'C0', alpha = 1, linewidth = 3, label = r'$P_W$')
    axd.plot(L[:,j],Xwt[:,j],'C3', alpha = 1, linewidth = 3, label = r'$B$')
    axd.plot(L[:,j],Zwt[:,j],'C0', alpha = 1, linewidth = 3, label = r'$W$')


    
    #axb.plot(L[:,j],Xwt[:,j],'g--', alpha = 1, linewidth = 2, label = r'$FC_W$')
    #axb.plot(L[:,j],Zwt[:,j],'y--', alpha = 1, linewidth = 2, label = r'$FC_W$')
    
    ax.plot(l,line,'k', alpha = 0.3, linewidth = 1)
    axb.plot(l,line,'k', alpha = 0.3, linewidth = 1)
    #ax.annotate(r'$\varepsilon_B = %.1f$' %em, (0.14, 2.6), fontsize = 15)


    ax.fill_between(l,line, facecolor='k',alpha = 0.1, interpolate=True)
    axb.fill_between(l,line, facecolor='k',alpha = 0.1, interpolate=True)
    ax.set_xscale("log")
    axb.set_yscale("log")
    axc.set_xscale("log")
    axc.set_yscale("log")
    axd.set_xscale("log")
    axd.set_yscale("log")
    ax.grid(alpha = 0.3, linewidth = 2)
    axb.grid(alpha = 0.3, linewidth = 2)
    axc.grid(alpha = 0.3, linewidth = 2)
    axd.grid(alpha = 0.3, linewidth = 2)
    axd.set_xlabel(r'$x$')
    ax.set_xlim([1e-1,1e1])
    ax.set_ylim([0,3])
    axb.set_xlim([1e-1,1e1])
    axb.set_ylim([1e-2,1e2])
    axc.set_xlim([1e-1,1e1])
    axc.set_ylim([1e-3,1e3])
    axd.set_xlim([1e-1,1e1])
    axd.set_ylim([1e0,1e5])
    
    print(sm[N-1],sw[N-1],smd[N-1],swd[N-1],smdd[N-1],swdd[N-1])
    
ax4.legend(loc = 'center left', bbox_to_anchor=(1.1, 0.525), fontsize = 15, ncol = 1)
ax9.legend(loc = 'center left', bbox_to_anchor=(1.1, 0.7), fontsize = 15, ncol = 1)
ax1.set_ylabel('promoter FC', fontsize = 17)
ax6.set_ylabel('protein FC', fontsize = 17)
ax10.set_ylabel('promoter activity', fontsize = 17)
ax14.set_ylabel('protein activity', fontsize = 17)

sm_exp = 1.85834610;
smd_exp = 0.535182514;
smdd_exp = 0.614542045;
sw_exp = 1.311341103;
swd_exp = 0.822331879;
swdd_exp = 1.345341568;

ax1.errorbar(1, sm_exp, yerr=0, fmt='sC3', mec='k', ecolor = 'k', ms = 7, capsize=0)
ax1.errorbar(1, smd_exp, yerr=0, fmt='sC3', mec='k', ecolor = 'k', ms = 7, capsize=0)
ax1.errorbar(1, smdd_exp, yerr=0, fmt='sC3', mec='k', ecolor = 'k', ms = 7,  capsize=0)
ax1.errorbar(1, sw_exp, yerr=0, fmt='sC0', mec='k', ecolor = 'k', ms = 7,  capsize=0)
ax1.errorbar(1, swd_exp, yerr=0, fmt='sC0', mec='k', ecolor = 'k', ms = 7,  capsize=0)
ax1.errorbar(1, swdd_exp, yerr=0, fmt='sC0', mec='k', ecolor = 'k', ms = 7,  capsize=0)


plt.savefig('bounds.pdf', transparent = True, bbox_inches='tight')

plt.show()

end = time.time()

print(end - start)
