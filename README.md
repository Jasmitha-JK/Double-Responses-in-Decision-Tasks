# Simulation-Based Inference of the Racing Diffusion Model using BayesFlow

A complete workflow for amortized Bayesian inference on a Racing Diffusion Model (RDM) with double-response monitoring. This project implements a neural posterior estimator in [BayesFlow](https://github.com/stefanradev93/BayesFlow) to recover latent decision-making parameters from behavioral data.

## What We Did

This is a course project for **Simulation-Based Inference (TU Dortmund, 2026)** implementing an end-to-end SBI workflow:

1. **Built a stochastic RDM simulator** that models two-choice decision-making with response-time data and double-response monitoring (when a participant makes a second, rapid response for the alternative)
2. **Wrapped the simulator for BayesFlow** with prior distributions over 7 latent parameters per participant
3. **Trained amortized neural posterior estimators** using DeepSet summary networks and FlowMatching inference networks, entirely on simulated data (50 epochs, ~8,000 synthetic participants, 3.2M synthetic trials)
4. **Validated the inference procedure** via convergence analysis, parameter-recovery sanity checks, and Simulation-Based Calibration (SBC) on 200 synthetic datasets
5. **Applied inference to real data** from Evans et al. (2020) — 25 participants, 400 trials each, 232 observed double responses
6. **Compared model predictions to reality** via posterior predictive checks across reaction time, accuracy, and double-response rate

**Key result:** BayesFlow successfully recovers parameters, but the basic RDM model architecture fails spectacularly at reproducing double responses (~83% predicted vs. ~2.3% observed), revealing that lateral inhibition or leakage mechanisms are necessary which is consistent with Evans et al.'s original findings.

## What's in the Report

The full technical report covers:

| Section | What | 
|---------|------|
| **Introduction** | Motivation: why double responses matter for evidence accumulation models, what Evans et al. (2020) showed, the RDM structure, and the inferential goal |
| **Data** | Evans dataset: 25 participants, 400 trials/participant, 10K total trials, 2.32% observed double-response rate |
| **Statistical Model** | RDM mechanics: two accumulators, threshold dynamics, double-response monitoring window, seven latent parameters with uniform priors |
| **Approximator** | Neural network architecture: DeepSet summary network + FlowMatching inference network |
| **Training** | Online simulation, 50 epochs × 20 batches × 8 participants = 8K datasets, training loss decay from 2.85 → 1.19 |
| **Diagnostics** | Three calibration checks: training convergence, parameter recovery on synthetic data, SBC rank histograms (200 datasets) |
| **Inference on Evans Data** | Posterior estimates for all 25 participants; posterior means + 95% credible intervals per parameter |
| **Posterior Predictive Validation** | Model simulation vs. observed data: Mean RT, accuracy, double-response rate reveals major model inadequacy |
| **Limitations & Improvements** | Why the RDM fails (no inhibition), next steps (LCA model), calibration caveats |
| **Conclusion** | Summary: SBI workflow works, inference is reasonable, but the cognitive model needs lateral inhibition |
| **Reflection** | Personal learnings from the project |
| **Appendix** | Participant-level validation metrics (CSV), parameter estimates (CSV), SBC output |

## Overview of the Approach

**The Problem**: Speeded two-choice decisions produce reaction times and choices, but sometimes participants make a second rapid response for the unchosen alternative ("double responding"). Evans et al. (2020) showed this phenomenon constrains evidence accumulation models.

**Our Solution**: We use simulation-based inference (SBI) specifically amortized Bayesian inference via neural networks to:
1. Skip expensive likelihood calculations by relying on simulation
2. Train a neural posterior estimator on simulated data once
3. Apply that trained estimator to any new participant instantly (amortization)

**Why SBI fits**: The RDM is trivial to simulate trial-by-trial, but there's no closed-form likelihood for the full observation process. SBI is the natural tool.

**The Workflow**:
```
Prior → RDM Simulator → Synthetic {params, behavior} pairs
                              ↓
                    DeepSet Summary Network
                              ↓
                    FlowMatching Posterior
                              ↓
                    [Trained once, reused for all 25 participants]
```

## Repository Structure

```
.
├── README.md                          # This file
├── FINAL_SBC_SBI.ipynb               # Main notebook: training, inference, diagnostics
├── simulator.py                       # RDM trial simulator (core logic)
├── bf_simulator.py                   # BayesFlow wrapper: prior + RDM integration
├── train_bayesflow.py                # Training loop: DeepSet + FlowMatching networks
├── results/
│   ├── participant_parameters.csv     # 25 × 7 posterior-mean estimates
│   ├── validation_results.csv         # Participant-level posterior predictive checks
│   ├── sbc_rank_histograms.png       # SBC diagnostics (200 synthetic datasets)
│   └── posterior_predictive_summary.png  # Observed vs. simulated metrics
└── data/
    ├── evans_raw.csv                 # Empirical behavioral data (25 participants)
    └── (or external link to Evans et al. 2020)
```

## Quick Start

### Requirements
- Python 3.8+
- PyTorch (for BayesFlow)
- NumPy, SciPy, Pandas
- BayesFlow (install via pip or conda)
- Jupyter (to run the notebook)

### Installation

```bash
git clone https://github.com/Jasmitha-JK/Double-Responses-in-Decision-Tasks.git
cd Double-Responses-in-Decision-Tasks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch bayesflow numpy scipy pandas jupyter matplotlib seaborn
```

### Running the Analysis

**Option 1: Run the complete notebook** (recommended)
```bash
jupyter notebook FINAL_SBC_SBI.ipynb
```
This walks through the entire pipeline: simulation, training, SBC, and inference on the Evans dataset.

**Option 2: Use individual modules**
```python
from simulator import RDM

# Initialize the model
model = RDM()

# Simulate one trial
parameters = {'A': 0.4, 'b': 1.0, 'vc': 1.5, 've': 3.0, 't0': 0.3, 'sv': 0.3, 'ster': 0.05}
trial = model.simulate_trial(parameters)

# Simulate 400 trials for one "participant"
participant_data = model.simulate_participant(parameters, n_trials=400)
```

## Methods

### Racing Diffusion Model

Two independent accumulators race toward a threshold. Evidence starts at a random initial value (uniform on [0, A]), evolves via Brownian motion with drift rates vc and ve, and the first to reach threshold B = A + b produces the first response. To capture double responses, evidence continues accumulating; if the losing accumulator reaches threshold within a monitoring window (~150 ms), a second response is recorded.

**Parameters** (seven latent variables per participant):
- **A**: Starting-point range (uniform prior: 0.20 – 0.60)
- **b**: Decision threshold component (0.40 – 1.20)
- **vc**: Drift rate for correct/first accumulator (0.50 – 4.00)
- **ve**: Drift rate for error/second accumulator (0.50 – 4.00)
- **t0**: Non-decision time (0.15 – 0.45 seconds)
- **sv**: Trial-to-trial drift variability (0.10 – 1.00)
- **ster**: Trial-to-trial non-decision-time variability (0.00 – 0.15)

### Neural Posterior Estimator

The BayesFlow workflow combines two neural networks:

1. **Summary Network (DeepSet)**: Pools 400 exchangeable trial observations into a fixed-size representation
2. **Inference Network (FlowMatching)**: Maps the summary to a posterior distribution over the 7 parameters

**Training details**:
- Online simulation: 50 epochs × 20 batches × 8 synthetic participants = ~8,000 simulated datasets
- ~3.2 million synthetic trials total
- Training loss decreased from ~2.85 to ~1.19

## Results

### Calibration (Simulation-Based Calibration)

SBC evaluates whether the trained posterior is well-calibrated on synthetic data. We simulated 200 independent datasets with known ground-truth parameters, ran inference, and recorded the rank of the true parameter among 500 posterior samples.

**Finding**: Rank histograms were broadly distributed but showed visible deviations from uniformity for some parameters, indicating reasonable but not perfect calibration.

### Inference on Evans et al. Dataset

Applied the trained network to 25 empirical participants. Posterior means:
- **A**: 0.410 ± 0.007 (range: 0.398–0.430)
- **b**: 1.036 ± 0.035 (range: 0.974–1.144)
- **vc**: 0.973 ± 0.014 (range: 0.933–0.994)
- **ve**: 3.595 ± 0.156 (range: 2.980–3.785)
- **t0**: 0.385 ± 0.013 (range: 0.359–0.425)
- **sv**: 0.461 ± 0.006 (range: 0.451–0.478)
- **ster**: 0.076 ± 0.001 (range: 0.072–0.079)

### Posterior Predictive Validation

Simulated 400 trials from each participant's posterior mean, compared with observed data:

| Metric | Observed | RDM Simulated | Error |
|--------|----------|---------------|-------|
| Mean RT | 0.435 s | 0.720 s | 0.285 s |
| Accuracy | 0.432 | 0.129 | 0.303 |
| Double-response rate | 2.32% | 82.88% | 80.56% |

**Interpretation**: The dramatic discrepancy in double-response rate is not an inference artifact (SBC shows the posterior is reasonably calibrated) but reflects the model's fundamental limitation. The RDM without lateral inhibition overpredicts double responses which is consistent with Evans et al.'s original finding that models with inhibition provide much better accounts of the data.

## Limitations & Next Steps

1. **Model architecture**: The RDM without lateral inhibition cannot explain the observed double-response rate. The next step is implementing a Leaky Competing Accumulator (LCA) model, repeating the same BayesFlow and diagnostic workflow.

2. **Posterior contraction**: A full posterior-contraction analysis (posterior variance / prior variance for each parameter) would quantify how much information the 400 trials per participant provide.

3. **Computational budget**: The SBC run used 200 synthetic datasets and 500 samples; larger studies would provide more stable diagnostics.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{Evans2020,
  author = {Evans, Nathan J. and Dutilh, Gilles and Wagenmakers, Eric-Jan and van der Maas, Han L. J.},
  title = {Double responding: {A} new constraint for models of speeded decision making},
  journal = {Cognitive Psychology},
  year = {2020},
  volume = {121},
  pages = {101292}
}

@article{Radev2023,
  author = {Radev, Stefan T. and Schmitt, Marvin and Schumacher, Lukas and others},
  title = {BayesFlow: Amortized {B}ayesian workflows with neural networks},
  journal = {Journal of Open Source Software},
  year = {2023},
  volume = {8},
  number = {89},
  pages = {5702}
}
```

## Authors

- Md. Rehan Ashraf Sharief
- Pankhuri Thakur
- Jasmitha Jagadeesh Karkera

**Course**: Simulation-Based Inference (TU Dortmund University, 2026)

## References

1. Evans, N. J., Dutilh, G., Wagenmakers, E.-J., & van der Maas, H. L. (2020). Double responding: A new constraint for models of speeded decision making. *Cognitive Psychology*, 121, 101292.
2. Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, A., Pratz, V., Schälte, Y., Köthe, U., & Bürkner, P.-C. (2023). BayesFlow: Amortized Bayesian workflows with neural networks. *Journal of Open Source Software*, 8(89), 5702.
3. Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: The leaky, competing accumulator model. *Psychological Review*, 108(3), 550–572.

## License

[Specify your license here, e.g., MIT, GPL-3.0, etc.]

---

**Questions?** Open an issue or contact the authors.
