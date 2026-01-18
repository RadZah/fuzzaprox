# Fuzzaprox


A Python package for Fuzzy approximation using F-transforms over residuated lattices.

## Installation

```bash
pip install fuzzaprox
pip install numpy
```



## Usage

```python
import numpy as np
from fuzzaprox import Fuzzaprox


# 0) Instantiate fuzzaprox
fa = Fuzzaprox()


# 1) Set input data to approximate
fa.set_input_data(np.array([1, 2, 3, 4, 5]))

# 2) Define and set FUZZY SETs shape
fa.define_fuzzy_set(base_start=0, kernel_start=12, kernel_end=14, base_end=26)

# 3) Run the approximation calculation
fa.run()

# RESULTS

# Forward approximations
approx_fw_upper = fa.get_fw_approx_upper()
approx_fw_bottom = fa.get_fw_approx_bottom()
approx_fw_upper_x = approx_fw_upper["fw_x"]
approx_fw_upper_y = approx_fw_upper["fw_y"]
approx_fw_bottom_x = approx_fw_bottom["fw_x"]
approx_fw_bottom_y = approx_fw_bottom["fw_y"]

# Inverse approximations
approx_inv_upper = fa.get_inv_approx_upper()  # get upper approximation
approx_inv_bottom = fa.get_inv_approx_bottom()  # get bottom approximation
```


# Plot Resulst
```python
import matplotlib.pyplot as plt

# Plot results
fig, axs = plt.subplots(2, figsize=(10, 7))

x_vals = fa.get_x_axes()
y_input_normalized_vals = fa.get_normalised_y_vals()


# Forward approximations
axs[0].plot(x_vals, y_input_normalized_vals)
axs[0].plot(approx_fw_upper_x, approx_fw_upper_y, marker='o', linestyle='None', color='r')
axs[0].plot(approx_fw_upper_x, approx_fw_bottom_y, marker='o', linestyle='None', color='b')

# Inverse approximations with Forward points
axs[1].plot(x_vals, y_input_normalized_vals)
axs[1].plot(x_vals, approx_inv_bottom, color='b')
axs[1].plot(x_vals, approx_inv_upper, color='r')

plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle

plt.show()
```


## License

See `LICENSE` file for license information.