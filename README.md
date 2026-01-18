# Fuzzaprox

## version fuzzaprox-002

A Python package for Fuzzy approximation using F-transforms over residuated lattices.

## Installation

Install the required dependencies:

```bash
pip install fuzzaprox
```

For example case we will need a numpy package as well
```bash
pip install numpy
```



## Usage

```python
import numpy as np
from fuzzaprox import Fuzzaprox as Fa

fa = Fa.Fuzzaprox()

# Create example input data as a list
y_vals = [1, 2, 3, 4, 5]
# set input data to approximate
fa.set_input_data(y_vals)

# Definition of fuzzy sets - fuzzy set shape
fa.define_fuzzy_set(base_start=0, kernel_start=12, kernel_end=14, base_end=26)

# Run the approximation
fa.run()

# Get results
approx_inv_upper = fa.get_inv_approx_upper()
approx_inv_bottom = fa.get_inv_approx_bottom()
approx_fw_upper = fa.get_fw_approx_upper()
approx_fw_bottom = fa.get_fw_approx_bottom()
```

## Requirements

See `requirementest.txt` for the list of dependencies.

## License

See `LICENCE` file for license information.