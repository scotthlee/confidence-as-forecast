# Lost Submarine Simulation

Simulation code accompanying **"Confidence as Forecast: A Decision-Theoretic Interpretation of Confidence Intervals"** (Lee, 2026), [arXiv:2602.15581](https://arxiv.org/abs/2602.15581).

## Overview

This script replicates and extends the lost-submarine thought experiment from [Morey et al. (2016)](https://doi.org/10.3758/s13423-015-0947-8). A rescue ship observes pairs of bubbles rising uniformly at random from a submarine's hull and uses them to construct confidence intervals for the location of the hatch. We then evaluate several coverage forecasting strategies — constant forecasts, width-conditional forecasts, and nesting-conditional forecasts — using Brier scores.

The main result: treating the confidence level as a probabilistic forecast for coverage, and refining that forecast with θ-free statistics like relative interval width, strictly improves predictive performance over both the "always covers" declaration and the constant design-level forecast.

## Requirements

- Python 3.8+
- NumPy
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install numpy pandas scikit-learn
```

## Usage

```bash
python morey_sim.py
```

This runs a grid of 110 simulation configurations (hatch location from 0 to 10, hull width from 10 to 110) with 100,000 bubble pairs each and prints summary tables matching Tables 1 and 2 in the paper.

## What the script does

1. **Constructs confidence intervals** under three 50% coverage procedures: nonparametric (NP), universally most powerful (UMP), and sampling-distribution (SD).

2. **Evaluates marginal forecasting strategies** for the NP and UMP procedures: always predicting coverage (q = 1), predicting at the design level (q = 0.5), and predicting from coverage probability conditioned on relative interval width.

3. **Evaluates joint forecasting strategies** for the SD + UMP pair: constant joint coverage forecast, coverage conditioned on nesting direction, and coverage conditioned on nesting direction plus outer interval width.

4. **Reports Brier score means and variances** across all configurations, along with marginal and joint coverage rates.

## Citation

```bibtex
@article{lee2026confidence,
  title={Confidence as Forecast: A Decision-Theoretic Interpretation of Confidence Intervals},
  author={Lee, Scott},
  journal={arXiv preprint arXiv:2602.15581},
  year={2026}
}
```

## See also

- [Either a Confidence Interval Covers, or It Doesn't (Or Does It?)](https://arxiv.org/abs/2602.15562) — companion paper on ex-post coverage probability.

## License

MIT
