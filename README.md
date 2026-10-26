# MSGT-Based AFG Micro-Column Buckling Analysis and Multi-Objective Inverse Design

[![Verification Suite](https://img.shields.io/badge/Verification%20Gate-V1--V7%20Passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official Python implementation and open replication package for the study:  
**"Pareto-Optimal Buckling Tailoring and Size-Dependent Inverse Design of Non-Uniform AFG Micro-Columns based on MSGT"**

---

## 🔬 Overview

This repository provides the complete, self-contained Python source code, solver verification suite, optimization algorithms, and post-processing tools to reproduce all numerical findings, tables, and figures presented in the manuscript.

### Core Features
- **Mechanical Formulation:** Bernoulli–Euler beam kinematics coupled with the Modified Strain Gradient Theory (MSGT), incorporating three material length scale parameters ($l_0, l_1, l_2$) to capture dilatation, stretch, and rotation strain gradients.
- **Continuous Parameterization:** Independent cubic B-spline curves governing cross-sectional width $b(x)$, thickness $h(x)$, and phase volume fraction $V_r(x)$ ($18$ continuous design variables).
- **High-Fidelity Solver:** Shifted Legendre polynomial Rayleigh–Ritz eigenvalue solver supporting classical (BC1) and higher-order (BC2) boundary conditions with weak-axis constraint enforcement ($b(x) \ge h(x)$).
- **Multi-Objective Framework:** Population-based metaheuristics including Lévy-flight Artificial Bee Colony with Adaptive Weighted-Sum Method (ABC-Levy + AWSM) and Non-Dominated Sorting Genetic Algorithm II (NSGA-II).
- **Verification Gate:** Automated 7-tier pre-optimization benchmark validation (V1–V7) matching published literature data to 4–5 decimal places.

---

## 📁 Repository Structure

```
.
├── msgt_ritz.py               # Core Rayleigh-Ritz MSGT buckling eigenvalue solver
├── bspline.py                 # Cubic B-spline geometry and material parameterization
├── materials.py               # Constitutive phase properties (Al/SiC & synthetic pair)
├── objectives.py              # Bi-objective function (mass vs P_cr) & constraint handler
├── section_model.py           # Column cross-section models and aspect ratio checks
├── reference_exact.py         # Exact rational quadrature reference benchmark (V4)
├── optimizers.py              # Metaheuristic search engines (ABC-Levy & NSGA-II)
├── awsm.py                    # Adaptive weighted-sum method with Pareto refinement
├── metrics.py                 # Multi-objective metrics (Hypervolume, IGD, Spacing)
├── stat_tests.py              # Omnibus Friedman, Nemenyi CD, and pairwise Wilcoxon tests
│
├── test_validation.py         # 7-tier literature verification suite (Tables 1-3 in [8, 9])
├── test_model.py              # Kinematic consistency, partition of unity, and bound tests
├── test_optimizers.py         # Optimization algorithm verification tests
├── VALIDATION_SPEC.md         # Formal mathematical specification of validation gate
│
├── run_campaign.py            # Master script to execute the 15-scenario optimization campaign
├── post_optimal.py            # Front aggregation, knee design extraction, and convex hulls
├── make_figures_results.py    # Generates Figures 2 through 8
├── generate_figure1.py        # Generates Figure 1 technical schematic
├── results_summary.py         # Summary table and statistics generator
├── results_manifest.py        # Verification manifest and cryptographic hash validator
│
├── results/
│   ├── post/                  # Post-processed Pareto fronts, knee designs, and ablation data
│   ├── three_algorithms/      # Multi-algorithm comparative matrices and runs
│   └── MANIFEST.csv           # File hashes and execution manifests
├── figures/                   # High-resolution vector/raster publication figures
├── requirements.txt           # Python library dependencies
└── LICENSE                    # Open-source MIT License
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the lightweight requirements:
```bash
git clone https://github.com/ibrahimaydogdu/MSGT_AFG_Beam_Buckling_OptimizationPublish.git
cd MSGT_AFG_Beam_Buckling_OptimizationPublish
pip install -r requirements.txt
```

### 2. Run the Verification Suite (7-Tier Gate)
Verify the Rayleigh–Ritz solver against published literature benchmarks:
```bash
pytest test_validation.py -v
```
Or execute directly to generate a detailed verification report:
```bash
python test_validation.py
```
*Expected result:* All 7 benchmark tiers pass with numerical agreement matching published literature to 4–5 decimal places (Legendre basis condition: $p \approx 10^{-15}$).

### 3. Run Model & Kinematic Tests
```bash
pytest test_model.py -v
```

---

## 📊 Reproducing Manuscript Figures and Tables

All publication figures and summary tables can be generated directly from the included post-processed results:

### Generate Figures 2–8
```bash
python make_figures_results.py
```
This regenerates:
- **Figure 2:** Multi-objective algorithmic distributions and Demšar critical difference rank diagrams.
- **Figure 3:** Local non-dominated envelope ratios across structural mass.
- **Figure 4:** Pooled empirical Pareto fronts $\mathcal{P}^*$ vs benchmark uniform/tapered columns.
- **Figure 5:** Size-effect progression across material length scale ratios $\mu \in \{0.0, 0.1, 0.2\}$.
- **Figure 6:** Dimensionless buckling parameter scaling across length scale sweeps.
- **Figure 7:** Optimal axial profiles ($b(x), h(x), V_r(x)$) for knee designs across 30 runs.
- **Figure 8:** Two-dimensional physical micro-column schematics and material contours.

### Generate Figure 1 Schematic
```bash
python generate_figure1.py
```

### Run Statistical Tests (Table 7)
```bash
python analyze_three.py
```

---

## ⚙️ Running Optimization Campaigns

To run a single optimization scenario (e.g., Al/SiC cantilever with $\mu = 0.1$ using NSGA-II):
```bash
python run_campaign.py --material ALSIC --bc CF --mu 0.1 --algorithm NSGA2 --runs 1
```

To execute the full 15-scenario multi-algorithm experimental campaign (30 independent runs per scenario, budget $B = 30{,}000$ evaluations):
```bash
python run_campaign.py --all
```

---

## 📜 Replication & AI Usage Disclosure

- **Replication:** This code repository is provided to ensure full transparency and scientific reproducibility.
- **AI Tool Usage:** Google Gemini Flash 3.8 and Anthropic Claude Sonnet 5.5 were utilized for code debugging, syntax checking, and language quality enhancement. All mathematical derivations, numerical formulations, and engineering interpretations were independently developed and approved by the authors.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
