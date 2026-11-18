# Safety-Aware Coordinated Reinforcement Learning for Electric Bus Fleet Charging

> **ICAMMA 2026 — IEEE International Conference**  
> Diya Kaushik · Jayanti Sinha · Dr. Suveg Moudgil  
> Dept. of Computer Science & Engineering, Galgotias University, Greater Noida, India

---

## Overview

This repository contains the full source code, simulation data, paper LaTeX source, and interactive dashboard for our ICAMMA 2026 paper on coordinated reinforcement learning for electric bus fleet charging.

We present a **tabular Q-learning framework** evaluated on the **official Guelph Transit GTFS feed** (Route 17, Ontario, Canada) using real **Nova Bus LFSe+** specifications. A hard SoC floor action mask guarantees zero emergency under-charge events, while a coordination layer routes buses to the least-congested charger — evaluated through a clean ablation that uses the exact same trained Q-table.

---

## Key Results

| Strategy | Charger Util. | Passengers | SoC Std Dev | Cost (CAD$) | Emergencies |
|---|---|---|---|---|---|
| Baseline | 0.599 ± 0.018 | 14,963 ± 389 | 4.3% | 121.5 ± 8.1 | 0 |
| Heuristic | 0.573 ± 0.017† | 14,797 ± 401 | 4.3% | **107.3 ± 7.2‡** | 0 |
| RL-Tabular | 0.654 ± 0.019‡ | 14,181 ± 348‡ | 2.9% | 119.2 ± 7.6 | 0 |
| **RL-Coordinated** | **0.665 ± 0.017‡** | 14,600 ± 336‡ | **2.1%** | 130.1 ± 8.3† | **0** |

†p<0.05, ‡p<0.01 vs Baseline (Mann-Whitney U, two-sided). 30 Monte Carlo runs per strategy.

**Headline numbers:**
- **+11.0%** charger utilisation improvement (p < 0.01)
- **98%** passenger throughput maintained
- **−51%** SoC variance reduction (Cohen's d = −0.80)
- **0** emergency under-charge events across all 120 run combinations
- Physically validated: 57–67% utilisation range for this bus/route config

---

## Repository Structure

```
├── code/
│   ├── simulation.py          # Full simulation: RL agent, env, Monte Carlo eval
│   └── generate_figures.py    # Reproduce all paper figures
├── paper/
│   ├── paper.tex              # LaTeX source (IEEEtran, compile on Overleaf)
│   └── paper.pdf              # Compiled paper
├── figures/
│   ├── fig0_route_map.png         # Guelph Transit Route 17 geometry
│   ├── fig1_training_convergence.png
│   ├── fig2_soc_comparison.png
│   ├── fig4_coordination_benefit.png
│   ├── fig5_statistical_significance.png
│   ├── fig6_charger_grid.png
│   ├── fig7_policy_analysis.png
│   ├── fig8_ablation.png
│   └── fig9_sensitivity.png
├── data/
│   ├── table1_main.csv            # Main results (30 MC runs, all strategies)
│   ├── table2_statistics.csv      # Statistical test results
│   ├── table3_sensitivity_fleet.csv
│   ├── table3_sensitivity_chargers.csv
│   └── table3_sensitivity_power.csv
├── dashboard/
│   └── index.html             # Interactive research dashboard (open in browser)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full simulation

```bash
python code/simulation.py
```

This runs 30 Monte Carlo trials for all 4 strategies and prints the results table. Runtime: ~3–5 minutes on a standard CPU.

### 3. Reproduce all figures

```bash
python code/generate_figures.py
```

Figures are saved to the `figures/` directory.

### 4. View the interactive dashboard

Open `dashboard/index.html` in any browser — no server needed, fully self-contained.

---

## System Design

### Real-World Data

| Parameter | Value | Source |
|---|---|---|
| Route | Route 17 Woodlawn–Watson | Official Guelph Transit GTFS |
| GTFS Shape ID | 8586 | Guelph Transit open data |
| Route length | 32.65 km, 71 stops | GTFS shape + RDP simplification |
| Bus battery | 564 kWh | Nova Bus LFSe+ spec |
| Bus range | 470 km | Nova Bus LFSe+ spec |
| Charger power | 150 kW × 4 stations | Guelph 2025 staff report |
| ON peak price | $0.170/kWh CAD | Ontario OEB 2024 |
| ON off-peak price | $0.082/kWh CAD | Ontario OEB 2024 |

### State Space

Each bus observes a 6-dimensional state at every 15-second timestep:

```
s = (s_soc, s_dist, s_time, s_pax, s_queue, s_util)
```

Discretised into **10 × 6 × 6 × 3 × 3 × 3 = 9,720 states** — 1.8× more efficient than the naive 51,200-state formulation (3.0% vs 1.4% coverage at the same training budget).

### Safety Constraint

A hard action mask forces charging when `SoC < 22%`, regardless of the Q-table policy. This **provably eliminates** emergency under-charge events — unlike reward shaping, which can fail under sparse coverage.

### Coordination Layer

RL-Coordinated uses the same trained Q-table as RL-Tabular, differing only in charger selection:

```
score(charger) = distance + 2 × queue_depth
```

This design isolates the coordination contribution from the charging policy — a controlled ablation.

---

## Training

```
Episodes:     1,500
α (learning rate):   0.15
γ (discount):        0.95
ε decay:      1.00 → 0.01 (analytical, exact)
Timestep:     15 seconds
```

---

## Physics Validation

A key contribution is demonstrating that prior work with >85% charger utilisation contains calibration errors. The correct range is derived from first principles:

```
564 kWh ÷ 1.2 kWh/km = 470 km range
35 km/h × 8 h = 280 km per shift
280 km × 1.2 kWh/km = 336 kWh = 60pp SoC drop per shift
→ Each bus charges ~once per shift
→ 10 buses, 4 chargers → 2.5:1 ratio → 40–70% expected utilisation
→ Baseline result of 0.599 confirms simulation is correctly calibrated ✓
```

---

## Citation

If you use this code or results in your work, please cite:

```bibtex
@inproceedings{kaushik2026safety,
  title     = {Safety-Aware Coordinated Reinforcement Learning for Electric Bus
               Fleet Charging with Passenger Service Preservation},
  author    = {Kaushik, Diya and Sinha, Jayanti and Moudgil, Suveg},
  booktitle = {Proceedings of the International Conference on Advanced Methods
               in Mechanics and Applied Sciences (ICAMMA)},
  year      = {2026},
  note      = {IEEE}
}
```

---

## License

This code is released for research reproducibility. The paper text and figures are copyright the authors and Galgotias University. The simulation code is released under the MIT License — see `LICENSE` for details.

---

## Contact

- **Diya Kaushik** — diyakaushik027@gmail.com
- **Jayanti Sinha** — sinhajayanti26@gmail.com
- **Dr. Suveg Moudgil** — suveg.moudgil@galgotiasuniversity.edu.in

*Dept. of Computer Science & Engineering, Galgotias University, Greater Noida, India*
