# Fuzzaprox


A Python package for Fuzzy approximation using F-transforms over residuated lattices.

## Installation

```bash
pip install fuzzaprox
```


## Usage

```python
import numpy as np
from fuzzaprox import Fuzzaprox


# 0) Instantiate fuzzaprox
fa = Fuzzaprox()

# 1) Set input data to approximate
y = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.4*np.random.default_rng(42).normal(size=100)

fa.set_input_data(y)  # Sets Input data


# 2) Define and set FUZZY SETs shape
fa.define_fuzzy_set(base_start=0, kernel_start=12, kernel_end=14, base_end=26)

# 3) Run the approximation calculation
res = fa.run()

# Output
inp = res.input_data  # res.input_data: normalized input x/y
fw = res.forward      # res.forward: forward approximation (x, upper_y, bottom_y)
inv = res.inverse     # res.inverse: inverse approximation (x, upper_y, bottom_y)


# 4) Optional: convert the approximations back to the original data scale
#    The results above are on the normalized [0,1] scale used internally.
fwDen = fa.denormalise(fw)   # ApproxResults with both y-series in original units
invDen = fa.denormalise(inv)

fa.denormalise(inv.upper_y)  # a plain array of y-values works as well
```


# Plot Results
```python
import matplotlib.pyplot as plt

# Plot results
fig, axs = plt.subplots(2, figsize=(10, 7))

# Forward points
axs[0].plot(inp.x, inp.normalized_y)
axs[0].plot(fw.x, fw.upper_y, marker='s', linestyle='None')
axs[0].plot(fw.x, fw.bottom_y, marker='s', linestyle='None')
axs[0].set_title("Forward Approximations")

# Inverse approximations
axs[1].plot(inp.x, inp.normalized_y)
axs[1].plot(inv.x, inv.upper_y)
axs[1].plot(inv.x, inv.bottom_y)
axs[1].set_title("Inverse Approximations")

# Add spacing between subplots to prevent title overlap
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
plt.show()
```


## History

Version 0.0.1 (September 2023) was written as part of a bachelor's thesis.
The original code is kept on the `archive/v0.0.1` branch (tag `v0.0.1`).
From 0.0.2 onwards the package was restructured to follow Python conventions.


## Acknowledgements

This package grew out of a bachelor's thesis supervised by
RNDr. Martina Daňková, Ph.D., at the Department of Informatics and Computers,
Faculty of Science, University of Ostrava, Czech Republic.


## License

See `LICENSE` file for license information.