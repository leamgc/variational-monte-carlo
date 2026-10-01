# Theory
### Background

In quantum mechanics, systems are described by a wavefunction $\Psi(\mathbf{R},t)$, which is in general complex. The square of the amplitude of this wavefunction, $|\Psi|^2 = \Psi \Psi^*$ gives the probability density of finding the system at position $\mathbf{R}$. 

The evolution of a quantum system is governed by the time-dependent Schrödinger equation:

$$ i\hbar \frac{\partial{\Psi(\mathbf{R},t)}}{\partial{t}} = \hat{H} \Psi(\mathbf{R},t) $$

where 

$$ \hat{H} = -\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{R},t) \quad(1) $$ 

is the Hamiltonian - an operator which corresponds to the total energy of the system. The first term corresponds to the kinetic energy operator and $V(\mathbf{R},t)$ is the potential energy of the system. 

For a special but common case of systems where the Hamiltonian is fixed with time (ie $V(\mathbf{R},t) = V(\mathbf{R})$), the solutions are of the form $\Psi(\mathbf{R}, t) = \psi(\mathbf{R})e^{-iEt/\hbar} $, where $\psi(\mathbf{R})$ satisfies the time-independent Schrödinger equation:
$$\hat{H}\psi(\mathbf{R}) = E \psi(\mathbf{R}) \quad (2) $$ 
where $E$ is the exact energy eigenvalue associated with $\psi(\mathbf{R})$. While $\Psi(\mathbf{R}, t)$ depends on time, the probability density $ |\Psi(\mathbf{R}, t)|^2 = \psi(\mathbf{R})^2$ is time-independent. Additionally since the time-varying complex phase $e^{-iEt/\hbar} $ is not measurable, for time-independent states, $\Psi$ can always be chosen as real.

### Variational Method

The exact form of $\Psi$ for a quantum system is found by solving the Schrödinger equation. However it is in general not possible to obtain analytical solutions with the exception of a select few systems such as the quantum harmonic oscillator. Various numerical methods are therefore necessary to study more complex quantum systems. One such method is Variational Monte Carlo, which allows one to find the ground state energy and wavefunction of a quantum system and hinges on the variational principle of quantum mechanics. 

The variational principle states that for any trial wavefunction $\Psi_{trial}$, the corresponding expectation value of the energy $ E_{trial} $ will always satisfy $E_{trial} \ge E_{0}$, where $E_{0}$ is the true ground state energy of the system, with equality if and only if $\Psi_{trial} = \Psi_{0} $ (up to a multiplicative constant). This turns the problem of finding the ground state into a variational problem over $E_{trial}$. An appropriate form for the trial ground state wavefunction $\Psi_{trial}$ is guessed, motivated by the system's physical properties, and associated parameters $\mathbf{\alpha}$ are varied to minimise the expectation value of the energy, $E_{trial}$. The minimum value for the energy obtained and the associated wavefunction - $E_{min},\ \Psi_{min}$ - are then the best approximation to the true ground state achievable within the chosen form of the trial wavefunction.

There is a range of commonly used minimisation algorithms for VMC, the choice between which depends mostly on the dimensionality and complexity of the trial wavefunction. In this case for the quantum harmonic oscillator, a simple linear gradient descent algorithm is used because the system's low parameter count and smooth energy landscape do not require complex/second-order optimization techniques. 
### Monte Carlo Integration 
Central to the application of the VMC method is the determination of $E_{trial}$, the expectation of the energy associated with each trial wavefunction $\Psi_{trial}$. This is calculated via:
$$E_{trial} = \frac{\langle \psi_{trial} | \hat{H} | \psi_{trial} \rangle}{\langle \psi_{trial} | \psi_{trial} \rangle} = \frac{\int d\mathbf{R}\, \psi_{trial}(\mathbf{R})^2 E_L(\mathbf{R})}{\int d\mathbf{R}\, \psi_{trial}(\mathbf{R})^2} = \int d\mathbf{R}\, \rho_{trial}(\mathbf{R}) E_L(\mathbf{R}) \quad(3)$$
where $\rho_{trial}(\mathbf{R}) = \frac{\psi_{trial}(\mathbf{R})^2}{\int d\mathbf{R}\, \psi_{trial}(\mathbf{R})^2} $ is the probability density function associated with the trial wavefunction $\psi_{trial}$ and $E_L(\mathbf{R})$ is the local energy of the trial wavefunction at a particular point, calculated via the TISE:

$$ E_L(\mathbf{R}) = \frac{\hat{H}\psi_{trial}(\mathbf{R})}{\psi_{trial}(\mathbf{R})} . \quad(4) $$

Evaluating equation (1) is usually performed using Monte Carlo integration, especially in a common case of systems where $\int d\mathbf{R}\, \psi_{trial}(\mathbf{R})^2$ is computationally heavy or intractable and hence the pdf $\rho_{trial}$ associated with the trial wavefunction is only known up to some normalising constant. Metropolis sampling is then used as a means to sample from such a distribution and an array of N samples $\mathbf{x} = (x_{1}, x_{2}, ... , x_{i}, ... , x_{N})$ is drawn from $\rho_{trial}$. Because the samples are already drawn according to $\rho_{trial}$, the probability weighting is implicit in the density of the sample points themselves, and so an estimate for $E_{trial}$ can be found simply as the unweighted mean of $E_L$ evaluated at each sample $x_i$:

$$ E_{trial} \approx \frac{1}{N}\sum_{i=1}^{N} E_L(x_i) .$$

### Metropolis Sampling
Metropolis sampling is a Markov Chain Monte Carlo method that generates random samples from a target distribution without requiring direct sampling or explicit normalisation of that distribution. It is commonly used in VMC methods, where the trial wavefunction (and hence its associated probability density $\rho_{trial}$) can be arbitrarily complex, and where, as mentioned above, the normalising integral is typically computationally intractable. 

Metropolis sampling builds up a chain of samples using an iterative process. At each step, a candidate point $x_{cand}$ is proposed by drawing a random point from a Gaussian centred on the previous point $x_i$, with some arbitrary width $\sigma_{proposal}$.
$$ x_{cand} \sim N(\mu = x_i, \sigma = \sigma_{proposal}). $$
The ratio of probability density under $\rho_{trial}$ at $x_{cand}$ and $x_i$ is calculated and the probability of accepting the move to $x_{cand}$ is $p_{accept}$:
$$ p_{accept} = \min(1\ , \frac{\rho_{trial}(x_{cand})}{\rho_{trial}(x_i)}) .$$
That is, moves to higher density regions are always accepted and moves to lower density regions are accepted with probability proportional to how much less probable they are than the current position $x_i$. If the move to $x_{cand}$ is rejected, new candidate points are generated until a move is successful. Since only the ratio $\frac{\rho_{trial}(x_{cand})}{\rho_{trial}(x_i)}$ ever enters the acceptance decision, any overall normalising constant in $\rho_{trial}$ cancels, meaning the chain can be constructed without ever needing $\rho_{trial}$ explicitly normalised, and therefore more generally allows for the sampling of probability distributions which are known up to only a multiplicative constant. After discarding an initial burn-in period, where the chain is still influenced by its arbitrary starting point, the resulting sequence of points is distributed according to $\rho_{trial}$. Here the choice of $\sigma_{proposal}$ is abitrary and only affects the acceptance rate of proposed candidate points. For best efficiency of the Metropoolis algorithm, we aim for an acceptance rate in the range 20-50%. 

# Quantum Harmonic Oscillator

### Theory 
As one of the few quantum systems which is analytically solvable, the quantum harmonic oscillator is commonly used to validate the VMC method since its result can be verified against the exact solution. Additionally, both its coordinate space and parameter space for the trial wavefunction are one dimensional. This simplicity makes it a natural starting point to develop the VMC procedure without the unessecary computational complexity introduced by higher dimensionality. 

The harmonic oscillator describes the position $x$ of a particle existing within a potential $V(x) = \frac{1}{2}m \omega^2 x^2$. Before examining the quantum counterpart it is helpful to consider the classical solution. Classically, the lowest energy state or 'ground state' analogue of this system is the solution where the particle remains at rest at the potential minimum with zero energy:
$$x_0(t) = 0 \ , \ E_0^{classical} = 0 .$$ 
However in the quantum case, the uncertainty principle prevents the particle from having both a precisely defined position $(x=0)$ and momentum $(p=0)$. Instead, the quantum ground state of the system must have a non-zero energy, which prevents the particle from being perfectly localized at the bottom of the potential well. The analytic solution to the problem via solving the Schrödinger equation yields:
$$ \psi_0 (x) = (\frac{m\omega}{\pi \hbar})^\frac{1}{4} e^{-\frac{m\omega}{2 \hbar} x^2} \ , \ E_0 = \frac{1}{2}\hbar \omega . $$

In this project I implement the VMC method to obtain these results numerically, developing the framework that extends to more complex quantum systems where analytic solutions are not available. 

### Implementation
The chosen form for the trial wavefunction is motivated by four physical requirements of the system; $\psi_{trial}$ must be normalisable (ie $\int_{-\infty}^{+\infty} \lvert\psi_{trial}^2\rvert ^2 \ dx$ is finite), $\psi_{trial}$ must be even since the potential $V(x) = \frac{1}{2}m \omega^2 x^2$ is even ($V(-x) = V(x)$), $\psi_{trial}$ must be peaked at the potential minimum $x=0$ (highest likelihood of observing the particle at point of lowest potential), and $\psi_{trial}$ must be nodeless (no points where $\psi_{trial}$ = 0). Motivated by these requirements, for the quantum harmonic oscillator the form:
$$ \psi_{trial}(x) = e^{-\alpha x^2} $$
is chosen. While in this case, the guessed trial form is equal to the true analytic solution, this equivalence is not typical - for most systems, no choice of trial wavefunction form will match the true wavefunction exactly. The VMC method still converges to the best approximation of the ground state achievable within the chosen functional form with with an unknown systematic error.
Substituting the potential $V(x) = \frac{1}{2}m \omega^2 x^2$ into equation (1), we obtain the Hamiltonian for the quantum harmonic oscillator:
$$ \hat{H}_{QHO} = -\frac{\hbar^2}{2m}\nabla^2 + \frac{1}{2}m \omega^2 x^2 . $$
Applying this hamiltonian to equation (3) with trial wavefunctions of form $ e^{-\alpha x^2}$, we obtain an expression for the local energy:
$$ E_{L, QHO}(x) = \frac{\hat{H}_{QHO}\psi_{trial}(x)}{\psi_{trial}(x)} = \frac{-\frac{\hbar^2}{2m} \frac{d^2}{dx^2}e^{-\alpha x^2} + \frac{1}{2}m \omega^2 x^2 e^{-\alpha x^2} } {e^{-\alpha x^2}} = \frac{\hbar^2 \alpha}{m} + (\frac{1}{2}m\omega^2 - \frac{2 \hbar^2 \alpha^2}{m})x^2 .$$
The probability density $\rho_{trial}$ can be found:
$$ \rho_{trial} = \frac{\psi_{trial}^2}{\int_{- \infty}^{+ \infty}\psi_{trial}^2 dx} = \frac{e^{-2 \alpha x^2}}{\int_{- \infty}^{+ \infty} e^{-2 \alpha x^2}dx} .$$
While in this case the normalising factor is analytically solveable, and the form of $\rho_{trial}$ is a gaussian and thereform simple to sample from using established algorithms, such as the Box-Muller transform via $\texttt{numpy.random.normal}$. However, for the purpose of demonstrating the VMC method, we set the normalising factor as an unknown constant $K$ and use the Metropolis algorithm to sample from 
$$ \rho_{trial} = \frac{e^{-2 \alpha x^2}}{K} . $$
## Results 
### 1. Metropolis Sampler
The Metropolis sampler produces points discributed according to the target distribution
$$ \rho_{trial} = \frac{e^{-2 \alpha x^2}}{K} . $$
Figure (1) shows a histogram of the sampled points at $\alpha = 0.5$ against the analytic form of $\rho_{trial}(x)$, confirming that the sampler reproduces $\rho_{trial}$.
<p align="center">
  <img src="plots/sampler_validation.png" width="70%"><br>
  <em>Figure 1: Metropolis samples against the analytic density with N = 200000 samples at α = 0.5.</em>
</p>

The sampler must function across the range of $\alpha$ encountered during optimisation and from an arbitrary starting point $x_0$. An key diagnostic for the sampler performance is the acceptance rate of the sampler, and should be in the range $20-50$% for optimal efficiency of the sampler. The acceptance rate is determined by the relationship between the width of the target distribution, in this case $\sigma_{target} = \frac{1}{2\sqrt{\alpha}}$, and the width of the proposal distribution, $\sigma_{proposal}$. To keep the acceptance rate constant and within this desirable range across the entire parameter space of $\alpha$, a scaling of 
$$\sigma_{proposal} = \frac{1.629}{\sqrt{\alpha}} $$ 
was used, with the factor 1.629 used as it was found to keep the sampler at an ideal acceptance of ~$35$% across the range of $\alpha$ explored by the minimiser, as shown in Figure (2). 
<p align="center">
  <img src="plots/acceptance_rate.png" width="70%"><br>
  <em>Figure 2: Acceptance rate of Metropolis sampler with proposal width ∝ 1/√α.</em>
</p>

Since the minimiser was found in practice to remain within roughly $0 < \alpha < 6$, the acceptance rate was tested only across this range rather than for arbitrarily large $\alpha$, however the $1/\sqrt{\alpha}$ scaling and constancy of the acceptance rate is expected to hold more generally.
<p align="center">
  <img src="plots/burn_in.png" width="70%"><br>
  <em>Figure 3: Metropolis chain trajectories at α = 0.5 initialised at x₀ = ±2 and ±6. All four converge to the same region within ~100 steps, comfortably inside the 1000 samples discarded as burn-in (shaded).</em>
</p>

A burn-in phase of $\texttt{burn\_in} = 1000$ samples is used to discard the initial portion of the chain, during which the sampled points are still influenced by the arbitrary choice of $x_0$ and do not yet represent $\rho_{trial}$. Since the chain is inexpensive to extend, this cut is chosen generously; Figure (3) shows chains from starting points $x_0 = \pm 2, \pm 6$, converging to the same region within $\lesssim 100$  steps, comfortably inside the discarded window. 

### 2. Uncertainty Estimation & Autocorrelation
Since each $E_{trial}$ value is the result of a Monte Carlo integration, it carries an inherent statistical uncertainty. The central limit theorem gives the uncertainty on the mean of $N$ samples as


$$\sigma[\bar{E}_L] = \frac{\mathrm{std}(E_L)}{\sqrt{N}},$$

however this expression assumes the samples are statistically independent. Successive Metropolis samples are not; each candidate point is proposed as a displacement from the previous one, so nearby samples in the chain are correlated and the effective number of independent samples is smaller than $M$. The expression above therefore underestimates the true uncertainty if N is taken just as the number of samples produced by the Metropolis sampler. To correct for this we must consider the autocorrelation time $\tau$ of the samples, which is the number of steps the chain must make before a sample becomes effectively independent of an earlier one.  