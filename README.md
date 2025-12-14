# CKM_LLG: Limits-to-Learning Gap and Lower Bounds on Out-of-Sample R²

## Overview

This Python package provides an implementation of the limits-to-learning gap (LLG) and lower bounds on out-of-sample $R^2$ for ridge regression, as developed in Chen, Kelly, and Malamud (2025).


## Installation

Clone the repository manually:

```bash
git clone https://github.com/czm319319/CKM_LLG.git
cd CKM_LLG
conda env create -n <YOUR_ENV_NAME> -f environment.yml
conda activate <YOUR_ENV_NAME>
pip install -e .
```

## Usage

Here is a basic example of how to use the package:

```python
import numpy as np
from src.LLG.llg import LLG_Bounds

seed = 123
rng = np.random.default_rng(seed)

# Dimensions
T_train = 500
T_test = 400
P = 800

sigma_eps = 1.0

# Design matrices
X_train = rng.standard_normal((T_train, P))
X_test = rng.standard_normal((T_test, P))

# True beta
beta_true = rng.standard_normal(P)
beta_true = beta_true / np.linalg.norm(beta_true)

# Outcomes
y_train = X_train @ beta_true + sigma_eps * rng.standard_normal(T_train)
y_test = X_test @ beta_true + sigma_eps * rng.standard_normal(T_test)

# Ridge grid
z_grid = np.logspace(-4, 1, 100)

######### LLG Bounds #########
model = LLG_Bounds(z_grid=z_grid, confidence_level=95).fit(X_train, y_train)

# predictions for each z 
yhat = model.predict(X_test)

# LLG(z)
llg = model.llg(X_test)

# (R2_OOS, asymptotic lower bound, one-side lower CI bound)
r2_oos, r2_lb, r2_ci_lower = model.r2_bounds(X_test, y_test)

print("LLG:", llg)
print("R2_OOS:", r2_oos)
print("Asymptotic LB:", r2_lb)
print("95% lower CI:", r2_ci_lower)


```

## Citation

If you use this package in your research, please cite the following paper:

> Limits To (Machine) Learning - Chen, Kelly, and Malamud (2025)
> 
> https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5909844

## License

This package is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you find bugs or have suggestions for improvements.

## Contact

For any inquiries, please reach out via [GitHub Issues](https://github.com/czm319319/Limits_to_Learning/issues) or email [your_email@example.com](mailto:your_email@example.com).
