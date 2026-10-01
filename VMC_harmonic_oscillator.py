#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:41:27 2026

@author: leagallien-currier
"""

import numpy as np
import matplotlib.pyplot as plt
import random

#TRIAL WAVEFUNCTION OF FORM e^-alpha r^2
def phi_trial(r, alpha):
    return np.exp(-alpha * r**2)

#PDF = phi^2 - for true pdf should have integrand normalising factor 
#however not needed here as ratio / comparing magnitudes only ever needed - avoids having to do integration 
#one of the uses of metropolis sampling/monte carlo integration as possible to sample from non-integrable pdfs. 
def rho(r, alpha):
    return phi_trial(r, alpha)**2

#Metropolis sampling - generate random samples drawn from target distribution.
def metropolis(alpha, x_0, sig, N, burn_in):
    samples = np.empty(N)
    samples[0] = x_0
    accepted = 0     

    for i in range(1, N): 
        x_prev = samples[i-1] #previous accepted sample
        
        #GENERATE CANDIDATE FOR NEXT SAMPLE -> sample from gaussian with mean = x_prev. Std of gaussian is tuned to control accept_rate
        x_cand = np.random.normal(x_prev, sig)
        
        #DECIDE IF X_CAND IS ACCEPTED
        
        if rho(x_cand, alpha) >= rho(x_prev, alpha):#x_cand exists in higher prob. region
            samples[i] = x_cand # ACCEPT MOVE -> x_cand always accepted if it exists in higher prob. region of pdf
            accepted += 1
            
        else: #x_cand exists in lower prob. region
            acceptance_prob = rho(x_cand, alpha)/rho(x_prev, alpha) # CALCULATE ACCEPTANCE PROBABILITY -> probability of accepting the move to lower prob. region = p(x_cand)/p(x_prev)
            
            if random.random() <= acceptance_prob: 
                samples[i] = x_cand #ACCEPT MOVE WITH PROBABILITY acceptance_prob
                accepted += 1
            else:
                samples[i] = x_prev 
                        
    accept_rate = accepted/(N-1) #CALCULATE ACCEPTANCE RATE OF CANDIDATE POINTS -> should aim for 20-50% for best efficiency. 
    samples_clean = samples[burn_in:] #REMOVE BURN IN PHASE -> initial points do not properly sample target pdf as are affected by choice of initial point x_0.
    
    return samples, samples_clean, accept_rate



#%%
#METROPOLIS BLOCKING -> method to calculate autocorrelation time and correction factor on uncertainty in E_{trial}. 

#Split generated metropolis samples into blocks size block_length and find the mean of these blocks. Blocked uncertainty is then the std of the block means/sqrt(number of blocks) Each block mean acts as an independent sample when block_length >> autocorrelation time (tau). 
def block_uncertainty(E_L_values, block_length): 
    n_blocks = len(E_L_values) // block_length
    E_L_trimmed = E_L_values[:n_blocks * block_length]  #discard remainder
    blocks = E_L_trimmed.reshape(n_blocks, block_length)
    block_means = blocks.mean(axis=1)
    return np.std(block_means) / np.sqrt(n_blocks)

running_sum = 0
for alpha_test in np.linspace(0.01, 6, num=100): #calculate average autocorrelation factor over domain of likely alpha values (although should be largely independent of alpha - this is because proposal width is scaled as 1/√α to hold the ratio of proposal width to distribution width fixed, which keeps both the acceptance rate and the autocorrelation time approximately constant for all alpha values.) Blocking is performed over range of alpha values just to get a better statistical estimate of correction factor. 
    samples = metropolis(alpha=alpha_test, x_0=0, sig=1.629/np.sqrt(alpha_test), N=20000, burn_in=1000)[1]  #Blocking analysis performed on particular set of metropolis samples at alpha = alpha_test. 
    std_vals = np.array([])
    block_length = np.array([])
    
    for i in range(1,1000): #Calculate the blocked uncertainty for a range of block lengths (1-1000)
        block_length = np.append(block_length, i)
        std_vals = np.append(std_vals, block_uncertainty(samples, i))
    
    plateau_error = np.mean(std_vals[15:64]) #plateau occurs between block size [16,64]. Calculate the mean corrected error of the plateau zone. 
    correction_factor = plateau_error/std_vals[0] #ratio of av corrected plateau error to naive error (block length 1)
    running_sum += correction_factor

AUTOCORR_FACTOR = running_sum/100 #Average autocorrealtion correction factor over 100 samples of varying alpha.

#Blocking analysis indicates the uncorrected uncertainty estimate underestimates σ by ~2.1× (AUTOCORR_FACTOR = 2.1), corresponding to an autocorrelation time of ~2.4 steps. 

#%%
#MINIMISATION ALGORITHM//MAIN CODE

#constants + tunable parameters
w = 1
m = 1
h_bar_2 = 1
x_0_metropolis = 0.0
current_num_samples = 20000
num_burn_in = 1000
alpha_i = 0.7
grad_descent_coeff = 0.1
max_iterations = 1500

#LOCAL ENERGY EQUATION -> specific local energy equation for the system, derived using time-independent Schrödinger equation and the system Hamiltonian. 
def E_L(x, alpha):
    return (h_bar_2)*alpha/(m) + (0.5*m*w**2 - 2*(h_bar_2)*(alpha)**2/m)*x**2

def E_v_estimator(alpha, num_samples):
    points_clean = metropolis(alpha=alpha, x_0=x_0_metropolis, sig=1.629/np.sqrt(alpha), N=num_samples, burn_in=num_burn_in)[1] #GENERATE POINTS SAMPLED FROM PSI^2 PDF -> metropolis algorithm generates array of points sampled from required distribution. 
    
    E_L_values = E_L(points_clean, alpha=alpha) #for each sampled point, calculate corresponding local energy. 
    
    E_v_estimate = np.mean(E_L_values) #CALCULATE VARIATIONAL ENERGY ESTIMATE OF TRIAL WAVEFUNCTION -> expectation value of local energies. Can simply calculate mean since points are sampled from distribution. 
    E_v_uncertainty = np.std(E_L_values)/(np.sqrt(num_samples-num_burn_in))*AUTOCORR_FACTOR #Uncertainty is estimated by treating samples as independent samples and accounting for autocorrelation between samples by scaling with autocorrelation factor - derived above using blocking method. 
    
    #Calculate gradient dE_dalpha using analytic VMC approach -> dE/dalpha = 2 ⟨ (E_L − ⟨E_L⟩) · ∂ln ψ/∂α ⟩, ∂ln ψ/∂α = -x^2 for ψ = e^(-alpha*x^2).
    g_i = -2*(E_L_values - E_v_estimate)*points_clean**2
    dE_dalpha = np.mean(g_i)
    dE_dalpha_uncertainty = np.std(g_i)/np.sqrt(num_samples-num_burn_in)
    
    return E_v_estimate, E_v_uncertainty, dE_dalpha, dE_dalpha_uncertainty
    

#Create arrays to store E_v and local gradient (dE_dalpha) values + uncertaintes.
E_v_history = np.array([])
E_v_uncertainty_history = np.array([])
dE_dalpha_history = np.array([])
dE_alpha_uncertainty_history = np.array([])
alpha_history = np.array([])    

for i in range(max_iterations):
    
    if len(E_v_history)<5:
        current_num_samples = 20000
    else:
        recent_spread = np.std(E_v_history[-5:])
        if recent_spread < 0.002:
            current_num_samples = 300000
            d_alpha = 0.025
        elif recent_spread < 0.01:
            current_num_samples = 50000
        else: 
            current_num_samples = 20000
    
    E_v_i = E_v_estimator(alpha=alpha_i, num_samples=current_num_samples)
    
    E_v_history, E_v_uncertainty_history, dE_dalpha_history, dE_alpha_uncertainty_history, alpha_history = np.append(E_v_history, E_v_i[0]), np.append(E_v_uncertainty_history, E_v_i[1]), np.append(dE_dalpha_history, E_v_i[2]), np.append(dE_alpha_uncertainty_history, E_v_i[3]), np.append(alpha_history, alpha_i) #RECORD VARIATIONAL ENERGY AND LOCAL GRADIENT (dE_dalpha) FOR EACH alpha VALUE
    
    alpha_i = alpha_i - grad_descent_coeff*E_v_i[2] #alpha UPDATE RULE -> linear gradient descent with static weighting coefficient
    
    if alpha_i <= 0: #ALPHA MUST BE NON-NEGATIVE -> set small positive floor to prevent negative alpha values
        alpha_i = 0.0001
        
    #DETERMINE IF ALGORITHM HAS CONVERGED TO A MINIMA 
    if len(alpha_history) >= 15 and np.all(np.abs(dE_dalpha_history[-15:]) < 1e-5):
        break


'''    
plt.errorbar(alpha_history, E_v_history, yerr=E_v_uncertainty_history)
plt.show()
plt.errorbar(alpha_history[-20:], E_v_history[-20:], yerr=E_v_uncertainty_history[-20:])
plt.show()
plt.errorbar(alpha_history[-20:], dE_dalpha_history[-20:], yerr=dE_alpha_uncertainty_history[-20:])
plt.show()
plt.plot(alpha_history)
'''
#%%

#CALCULATE FINAL E_0 and alpha_0 values from final window of converged sequence
window = 10
E_f = E_v_history[-window:]
E_error_f = E_v_uncertainty_history[-window:]
dE_dalpha_f = dE_dalpha_history[-window:]
dE_dalpha_error_f = dE_alpha_uncertainty_history[-window:]
alpha_f = alpha_history[-window:]

#ALPHA_0 FINAL VALUE
#Two sources of uncertainty in alpha_0 - 1. systematic offset, 2. statistical error
#1. Systematic Offset -> Alpha update rule converges towards true minima alpha_0 in geometric progression. Taylor expansion of E(alpha) allows us to approximate the remaining systematic offset between the final value of the sequence and the true minimum alpha_0 : E'(alpha) ≈ E''(alpha_0) (alpha - alpha_0). Calculate E''(alpha_0) by plotting dE_dalpha against alpha near minima and find gradient of linear fit. Remaining offset is given by E'(alpha)/E''(alpha_0)

E_second_0 = np.polyfit(alpha_f, dE_dalpha_f, 1)[0] #E''(alpha_0)
alpha_offset = dE_dalpha_f[-1]/E_second_0

#2. Statistical  error -> gradients dE_dalpha are monte carlo estimates and have statistical noise. A random error delta_g in a gradient value displaces alpha_0 by delta_g/E''(alpha_0). One - sigma variation for delta_g is estimated by the mean of the random errors on the final window of dE_dalpha values. 
g_sig = np.mean(dE_dalpha_error_f)
alpha_sig = g_sig/E_second_0 #Statistical error associated with alpha_0 value. 

alpha_0 = alpha_history[-1] - alpha_offset #Since we know the magnitude and sign of our systematic offset it can be subtracted from the last alpha value to find the true minima alpha_0.

#E_0 FINAL VALUE
#run final high sample monte-carlo integration at the computed alpha_0
E_0, E_0_sig, _, _ = E_v_estimator(alpha=alpha_0, num_samples=5000000)

print('Results:')
print(f'alpha_0 = 0.5 + ({alpha_0-0.5:.2e}),  sigma = {alpha_sig:.2e}  '
      f'({abs(alpha_0-0.5)/alpha_sig:.2f} sigma)')
print(f'E_0     = 0.5 + ({E_0-0.5:.2e}),  sigma = {E_0_sig:.2e}  '
      f'({abs(E_0-0.5)/E_0_sig:.2f} sigma)')


#%%

'''
Noticed that alpha=2 and sig=1.2 acheived acceptance rate ~30%. 
Desired rate is 20-50% for metropolis sampling -> Here tested two proposed sigma-alpha relationships (either sig ~ 1/alpha or sig ~ 1/sqrt(alpha) to hopefully keep the acceptance rate constant despite alpha varying in the algorith. 
The test below confirmed that the relationship sig = 1.629/sqrt(alpha) keeps the proposed sample acceptance rate in metropolis algorithm at ~desired 30%. 
'''

for x in [0.5,1,2,5,10]:
    l, points, rate = metropolis(alpha=x, x_0=0, sig=0.6*x, N=5000, burn_in=1000)
    l2, points2, rate2 = metropolis(alpha=x, x_0=0, sig=1.629/np.sqrt(x), N=5000, burn_in=1000)
    print(rate, rate2)


#%%
# --- PLOTS ----
# --- SHARED PLOT STYLING ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.frameon': False,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

FIGSIZE = (6, 4)
C_DATA, C_THEORY, C_REF = 'tab:blue', 'k', 'tab:red'
# 1. Metropolis Sampler 
# 1.1 Metropolis sampler against target distribution
alpha_demo = 0.5
_, s_clean, _ = metropolis(alpha=alpha_demo, x_0=0.0,
                           sig=1.629/np.sqrt(alpha_demo),
                           N=200000, burn_in=1000)

x_grid = np.linspace(-3.5, 3.5, 500)
rho_exact = np.sqrt(2*alpha_demo/np.pi) * np.exp(-2*alpha_demo*x_grid**2)

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.hist(s_clean, bins=120, density=True, color=C_DATA, alpha=0.45,
        edgecolor='none', label='Metropolis samples')
ax.plot(x_grid, rho_exact, color=C_THEORY, lw=1.4,
        label=r'$\rho_{trial}(x)=\sqrt{2\alpha/\pi}\,e^{-2\alpha x^{2}}$')

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'Probability density')
ax.set_title(rf'Sampled distribution vs. analytic density ($\alpha={alpha_demo}$)')
ax.legend()
fig.savefig('Plots/sampler_validation.png')
plt.show()

# 1.2 Acceptance rate of sampler vs alpha (1/sqrt(alpha) justification)
alpha_scan = np.linspace(0.05, 6.0, 40)
rates = np.array([metropolis(alpha=a, x_0=0.0, sig=1.629/np.sqrt(a),
                             N=20000, burn_in=1000)[2] for a in alpha_scan])

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.axhspan(0.20, 0.50, color='tab:green', alpha=0.12,
           label='Target range (20–50%)')
ax.plot(alpha_scan, rates, 'o-', color=C_DATA, ms=3.5, lw=1.0,
        label=r'$\sigma_{prop}=1.629/\sqrt{\alpha}$')

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'Acceptance rate')
ax.set_ylim(0, 1)
ax.set_title(r'Acceptance rate under $1/\sqrt{\alpha}$ proposal scaling')
ax.legend()
fig.savefig('Plots/acceptance_rate.png')
plt.show()

# 1.3 Burn in phase justification
alpha_demo = 0.5
n_chains, n_show = 4, 3000
starts = [-6.0, -2.0, 2.0, 6.0]

fig, ax = plt.subplots(figsize=FIGSIZE)
for x0 in starts:
    raw, _, _ = metropolis(alpha=alpha_demo, x_0=x0,
                           sig=1.629/np.sqrt(alpha_demo),
                           N=n_show, burn_in=0)
    ax.plot(raw, lw=0.6, alpha=0.8, label=rf'$x_0={x0:+.0f}$')

ax.axvline(num_burn_in, color=C_REF, ls='--', lw=1.2,
           label=f'Burn-in cut ({num_burn_in})')
ax.axvspan(0, num_burn_in, color=C_REF, alpha=0.07)

ax.set_xlabel(r'Step')
ax.set_ylabel(r'$x$')
ax.set_title('Chain equilibration from different starting points')
ax.legend(ncol=2, fontsize=8)
fig.savefig('Plots/burn_in.png')
plt.show()

# 2. Uncertainty Estimation + Autocorrelation 
# 2.1. Blocking analysis graph
# (uses std_vals and block_length from last pass of the AUTOCORR_FACTOR loop above)

log2_L = np.log2(block_length)

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.axvspan(log2_L.min(), 4, color='tab:orange', alpha=0.10)
ax.axvspan(4, 6, color='tab:green', alpha=0.15)
ax.axvspan(6, log2_L.max(), color=C_REF, alpha=0.08)
ax.axvline(4, color='0.4', ls='--', lw=0.9)
ax.axvline(6, color='0.4', ls='--', lw=0.9)

ax.plot(log2_L, std_vals, color=C_DATA, lw=1.1)
ax.axhline(plateau_error, color='tab:green', ls=':', lw=1.3,
           label=rf'$\sigma_{{plateau}}={plateau_error:.4f}$')
ax.axhline(std_vals[0], color='0.35', ls=':', lw=1.3,
           label=rf'$\sigma_{{naive}}={std_vals[0]:.4f}$')

y0, y1 = ax.get_ylim()
top = y0 + 0.93*(y1 - y0)
for xpos, txt in [(2.0, 'Correlated\n(underestimates $\\sigma$)'),
                  (5.0, 'Plateau\n($L \\gg \\tau$)'),
                  (7.7, 'Noise-dominated\n(too few blocks)')]:
    ax.text(xpos, top, txt, ha='center', va='top', fontsize=8, color='0.25')

ax.set_xlabel(r'$\log_2(\mathrm{block\ length})$')
ax.set_ylabel(r'Blocked uncertainty on $E_{trial}$')
ax.set_title('Blocking analysis: autocorrelation-corrected uncertainty')
ax.legend(loc='lower right')
fig.savefig('Plots/blocking_analysis.png')
plt.show()

# 3. Variational energy across parameter space
alpha_scan = np.linspace(0.15, 1.6, 30)
E_scan, E_err_scan, std_scan = [], [], []

for a in alpha_scan:
    _, pts, _ = metropolis(alpha=a, x_0=0.0, sig=1.629/np.sqrt(a),
                           N=40000, burn_in=num_burn_in)
    vals = E_L(pts, alpha=a)
    E_scan.append(np.mean(vals))
    E_err_scan.append(np.std(vals)/np.sqrt(len(vals)) * AUTOCORR_FACTOR)
    std_scan.append(np.std(vals))

E_scan = np.array(E_scan); E_err_scan = np.array(E_err_scan)
std_scan = np.array(std_scan)

# 3.1 E_v against alpha
a_fine = np.linspace(0.15, 1.6, 400)
E_analytic = h_bar_2*a_fine/(2*m) + m*w**2/(8*a_fine)

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(a_fine, E_analytic, color=C_THEORY, lw=1.3,
        label=r'Analytic $E_v(\alpha)=\frac{\alpha}{2}+\frac{1}{8\alpha}$')
ax.errorbar(alpha_scan, E_scan, yerr=E_err_scan, fmt='o', ms=3.5,
            color=C_DATA, capsize=2, lw=0.9, label='VMC estimate')
ax.axvline(0.5, color=C_REF, ls='--', lw=1.0, alpha=0.7)
ax.axhline(0.5, color=C_REF, ls='--', lw=1.0, alpha=0.7)
ax.plot(0.5, 0.5, '*', color=C_REF, ms=11,
        label=r'Exact: $\alpha_0=0.5,\ E_0=0.5$')

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$E_{trial}$')
ax.set_title('Variational energy across parameter space')
ax.legend()
fig.savefig('Plots/energy_vs_alpha.png')
plt.show()

# 3.2 Zero-variance property
sigma_analytic = np.abs(0.5*m*w**2 - 2*h_bar_2*a_fine**2/m) / (2*np.sqrt(2)*a_fine)

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(a_fine, sigma_analytic, color=C_THEORY, lw=1.3,
        label=r'Analytic $\mathrm{std}[E_L]$')
ax.plot(alpha_scan, std_scan, 'o', ms=3.5, color=C_DATA,
        label='Sampled')
ax.axvline(0.5, color=C_REF, ls='--', lw=1.0, alpha=0.7,
           label=r'$\alpha_0=m\omega/2\hbar$')

ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\mathrm{std}\,[E_L(x)]$')
ax.set_title('Zero-variance property at the optimal parameter')
ax.legend()
fig.savefig('Plots/zero_variance.png')
plt.show()

# 4. Optimisation and convergence
# 4.1 Minimisation trajectory (E(alpha) against iteration)& (alpha against iteration)
it = np.arange(len(alpha_history))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 5.4), sharex=True)

ax1.plot(it, alpha_history, color=C_DATA, lw=1.1)
ax1.axhline(0.5, color=C_REF, ls='--', lw=1.0, label=r'Exact $\alpha_0=0.5$')
ax1.set_ylabel(r'$\alpha$')
ax1.legend()
ax1.set_title('Gradient-descent optimisation trajectory')

ax2.errorbar(it, E_v_history, yerr=E_v_uncertainty_history,
             color=C_DATA, lw=1.0, elinewidth=0.6, capsize=1.5)
ax2.axhline(0.5, color=C_REF, ls='--', lw=1.0, label=r'Exact $E_0=0.5$')
ax2.set_xlabel(r'Iteration')
ax2.set_ylabel(r'$E_{trial}$')
ax2.legend()

fig.tight_layout()
fig.savefig('Plots/optimisation_trajectory.png')
plt.show()


