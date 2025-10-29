import numpy as np
from tqdm import tqdm

# Update to CODATA 2018 value (matches LaTeX.txt)
alpha_inv = 137.035999084
tol       = 1e-8
N         = 1_000_000_000

# Function for Pi_f(e) based on Ramanujan's first approximation (Section 1.1)
def pi_f(e):
    """Calculates the scale-invariant ratio Pi_f(e) = 2 / (pi * f(e))."""
    if e >= 1:
        return 2 / (np.pi * (3 - np.sqrt(3)))
    
    sqrt_term = np.sqrt(1 - e**2)
    
    f_e = (3 * (1 + sqrt_term) - 
           np.sqrt((3 + sqrt_term) * (1 + 3 * sqrt_term)))
    
    if f_e == 0:
        return 0
    
    return 2 / (np.pi * f_e)

# ---- Importance Sampling Parameters ------------------------------------
sigma_e = 1e-5
sigma_a = 1e-4
e0 = 0.908700001
a0 = 29.50374881 # Use the value from the paper's older precision for sampling center

matches = 0
for _ in tqdm(range(N), desc="Importance MC (Dual Check)"):
    # 1. Importance Sample e and a
    e = np.random.normal(e0, sigma_e)
    a = np.random.normal(a0, sigma_a)
    
    # Reject unphysical or unstable eccentricities
    if e <= 0 or e >= 1:               
        continue
    
    # 2. **CRITICAL FIX: Check Geometric Constraint (The Pi_f = 1/3 part)**
    pi_e = pi_f(e)
    if abs(pi_e - (1/3)) > tol:
        continue # Reject samples that don't satisfy the geometric constraint

    # 3. **CRITICAL FIX: Correct Calculation of Semi-Minor Axis b**
    # b = a * sqrt(1-e^2)  (Semi-minor axis)
    b = a * np.sqrt(1-e**2) 
    
    # Calculate perimeter P using Ramanujan's second approximation
    h = ((a-b)/(a+b))**2
    P = np.pi * (a+b) * (1 + 3*h/(10 + np.sqrt(4-3*h)))
    
    # 4. Check Physical Constant Correspondence (The P = alpha^-1 part)
    if abs(P - alpha_inv) < tol:
        matches += 1

print(f"Matches: {matches:,} / {N:,}  →  p ≈ {matches/N:.2e}")