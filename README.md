# Deep LPPLS Calibration with Neural Networks (and a Self‑Attention Baseline)

This repository accompanies the study **“Self‑attention mecanisms and the use of Neural Network for calibrating non‑linear equations”** (Alexandre Remiat, Jules Mourgues‑Haroche, Université Paris Dauphine, July 2025). It benchmarks neural‑network calibrators of the **Log‑Periodic Power Law Singularity (LPPLS)** model against the classical **Nelder–Mead** optimizer on WTI crude oil, Bitcoin and the S&P 500. 

---

## TL;DR

- **Problem.** Calibrate the nonlinear LPPLS model to detect bubble turning points (critical time \(t_c\)).
- **Approach.** Compare **MLP (MLNN)**, **RNN (LSTM)**, and **CNN** regressors—each optionally **augmented with one MHSA (multi‑head self‑attention) block**—to **Nelder–Mead**. 
- **Key result.** Plain neural models **outperform Nelder–Mead**, cutting mean timing error by **2–5×** and shrinking dispersion by ~**30%**, while **doubling/tripling** the share of admissible fits; **adding a single MHSA block does not reliably improve accuracy**. 

---

## Background: LPPLS in a Nutshell

LPPLS models bubble dynamics with a power‑law trend plus log‑periodic oscillations, targeting the **critical time** \(t_c\) when the bubble ends. In practice, three **nonlinear** parameters \((t_c,\omega,\alpha)\) and four **linear** parameters \((A,B,C_1,C_2)\) are estimated; linear parameters are slaved to the nonlinear triplet via least squares. 

To assess validity, fits are filtered with a **Lomb periodogram** on the de‑trended residuals to ensure oscillatory structure consistent with the optimized \(\omega\). Only fits passing this filter are counted as **admissible** (used in “confidence”).

---

## Data

- **WTI spot** (daily), Jan‑1986 → Dec‑2024, from **U.S. EIA**; selected windows include regimes with notable turning points (e.g., **2008‑07‑03**, **2011‑04‑29**, **2016‑02‑11**). 
- **Bitcoin** (daily) and **S&P 500** (daily) from Yahoo Finance, with experiment windows in **2013–2018** and **2018–2022** (BTC) and **2013–2018**, **2018–2022** (S&P 500). 

Time series are segmented into **overlapping sub‑intervals** via a sliding‑window scheme (backward end‑dates stepped weekly; start‑dates stepped proportionally to 75% of span / 15 weeks).

---

## Methods Overview

- **Baselines:** Nelder–Mead simplex on LPPLS with linear–nonlinear separation.
- **Neural calibrators:**  
  - **MLNN:** two hidden layers (ReLU), predicting \((t_c,\alpha,\omega)\); trained with MSE on \(\log p(t)\) reconstruction. 
  - **RNN (LSTM):** final hidden state → FC layer → \((t_c,\alpha,\omega)\). 
  - **CNN:** 1D conv trunk + adaptive avg‑pool → FC → \((t_c,\alpha,\omega)\).
  - **+ MHSA (optional):** single multi‑head self‑attention block inserted on top of the trunk, with residual & layer‑norm, then pooled to predict the parameters. 

- **Training / early stopping:** validation loss typically plateaus ≈ **40 epochs**; early stopping with **patience = 40**. 

- **Parameter bounds (daily):** \(t_c \in [0, 4\ \text{years}]\), \(\omega \in [6, 13]\), \(\alpha \in [0.1, 1]\). 

- **Metrics:** mean signed error \( \overline{t_c - t_c^{\text{true}}} \), std. dev. of error, and **Confidence** = fraction of fits passing the Lomb filter. 

---

## Results Highlights

- Across **WTI, BTC, S&P 500**, **MLNN/RNN/CNN** consistently **beat Nelder–Mead**: mean timing bias drops from **hundreds of days** (e.g., ~**+713** on WTI) to **sub‑100‑day** levels (often **<20 days** on WTI/SPX), with **~30% lower** dispersion and **2–3×** higher confidence.
- Adding a **single MHSA block** does **not** systematically help: it sometimes reduces bias slightly but **lowers confidence** and **increases volatility** in errors across datasets.

> See the paper’s WTI density plots and Tables 1–6 for per‑window summaries (e.g., 2003–2008, 2007–2011, 2011–2015 for WTI; 2013–2018 and 2018–2022 for BTC; 2013–2018 and 2018–2022 for S&P 500). 

---


## Suggested Repository Layout

```
.
├── README.md
├── data/
│   ├── raw/                # WTI, BTC, SP500 (daily)
│   └── processed/
├── src/
│   ├── lppls/              # LPPLS utilities (linear-nonlinear split, reconstruction)
│   ├── models/             # MLNN, RNN (LSTM), CNN, and *WithAttention variants
│   ├── training/           # loops, early stopping, metrics
│   └── evaluation/         # Lomb filter, confidence, error stats, tables
├── experiments/
│   ├── wti_2003_2008.yaml
│   ├── btc_2013_2018.yaml
│   └── spx_2018_2022.yaml
├── results/
│   ├── tables/
│   └── figures/            # density plots, loss curves
└── notebooks/
```

This layout mirrors the paper’s pipeline (segmentation → training → Lomb filtering → scoring). 

---

## How to Cite

If you use this repository or the reported results, please cite:

> **Alexandre Remiat and Jules Mourgues‑Haroche.** *Self‑attention mecanisms and the use of Neural Network for calibrating non‑linear equations.* Université Paris Dauphine, July 2025. 

**BibTeX**
```bibtex
@techreport{RemiatMourguesHaroche2025,
  title        = {Self-attention mecanisms and the use of Neural Network for calibrating non-linear equations},
  author       = {Alexandre Remiat and Jules Mourgues-Haroche},
  institution  = {Universit{\'e} Paris Dauphine},
  year         = {2025},
  month        = {July}
}
```

---

## License

Add your license of choice here (e.g., MIT, Apache 2.0).

---

## Contact

For questions about the paper or implementation details, please contact the authors. 
