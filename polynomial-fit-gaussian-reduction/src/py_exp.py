import numpy as np
import time

# Generate synthetic data
x = np.arange(1000, dtype=float)
y = 1 + 2*x + 3*x**2  # true model: y = 1 + 2x + 3x^2

degree = 2

# Time the polynomial fit
start = time.perf_counter()
coeffs = np.polyfit(x, y, degree)
end = time.perf_counter()

print("Python fit coefficients:", coeffs)
print(f"Python fit time: {end - start:.6f} seconds")
