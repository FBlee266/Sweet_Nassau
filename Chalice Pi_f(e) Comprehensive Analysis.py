import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from tqdm import tqdm
import os
import sys

# Set mpmath precision for high accuracy
mp.dps = 35

# --- Configuration & Data Loading ---
OUTPUT_DIR = "comprehensive_chalice_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the imaginary parts of the zeros
try:
    # Please ensure you have the 'riemann_zeros_2M.npy' file from previous steps
    IMAGINARY_ZEROS = np.load('riemann_zeros_2M.npy')
except FileNotFoundError:
    print("Error: riemann_zeros_2M.npy not found. Please ensure it is in the same directory.")
    sys.exit()

NUM_ZEROS_TO_ANALYZE = 2000000
IMAGINARY_ZEROS_SUBSET = IMAGINARY_ZEROS[:NUM_ZEROS_TO_ANALYZE]

# The sigma values to test, ranging from 0 to 1 with finer steps
TEST_SIGMA_VALUES = np.arange(0.0, 2.0, 0.01)

# --- Geometric Helper Functions ---
def calculate_distance_difference(point_comp, foci):
    """Calculates the absolute difference between distances from a point to two foci."""
    p = mp.mpc(point_comp.real, point_comp.imag)
    f1 = foci[0]
    f2 = foci[1]
    d1 = mp.fabs(p - f1)
    d2 = mp.fabs(p - f2)
    return mp.fabs(d1 - d2)

def calculate_eccentricity(sigma, t_n):
    """Calculates the eccentricity of the ellipse for a given sigma and zero."""
    sigma_mp = mp.mpf(str(sigma))
    t_n_mp = mp.mpf(str(t_n))
    denominator = mp.sqrt((sigma_mp - mp.mpf('0.5'))**2 + t_n_mp**2)
    numerator = mp.fabs(sigma_mp - mp.mpf('0.5'))
    if denominator > 0:
        return numerator / denominator
    return mp.nan

def pi_f(e):
    """Calculates the Pi_f(e) metric based on eccentricity e."""
    e = mp.mpf(e)
    if e < 0 or e >= 1:
        return mp.nan
    f_e = 3 * (1 + e) - mp.sqrt((3 + e) * (1 + 3 * e))
    return 2 / (mp.pi * f_e)

# --- Main Analysis Loop ---
def main():
    print("--- Running Comprehensive Chalice Analysis ---")

    # Storage for all three metrics
    mean_dist_diff_values = []
    std_dist_diff_values = []
    
    mean_e_values = []
    std_e_values = []

    mean_pif_e_values = []
    std_pif_e_values = []
    
    sigma_values_for_plot = []

    for sigma_val in tqdm(TEST_SIGMA_VALUES, desc="Processing sigma values"):
        dist_diff_for_current_sigma = []
        e_for_current_sigma = []
        pif_e_for_current_sigma = []
        
        # Foci for the current sigma value
        foci = [mp.mpc(mp.mpf(str(sigma_val)), 0), mp.mpc(mp.mpf(str(1 - sigma_val)), 0)]
        
        for t_n in IMAGINARY_ZEROS_SUBSET:
            test_point = mp.mpc(mp.mpf('0.5'), mp.mpf(str(t_n)))
            
            # 1. Calculate the distance difference metric
            dist_diff = calculate_distance_difference(test_point, foci)
            dist_diff_for_current_sigma.append(float(dist_diff))
            
            # 2. Calculate the eccentricity metric
            eccentricity = calculate_eccentricity(sigma_val, t_n)
            if not mp.isnan(eccentricity):
                e_for_current_sigma.append(float(eccentricity))
                
                # 3. Calculate the Pi_f(e) metric
                pif_e_val = pi_f(eccentricity)
                if not mp.isnan(pif_e_val):
                    pif_e_for_current_sigma.append(float(pif_e_val))

        # Calculate statistics for all three metrics
        if dist_diff_for_current_sigma:
            mean_dist_diff_values.append(np.mean(dist_diff_for_current_sigma))
            std_dist_diff_values.append(np.std(dist_diff_for_current_sigma))
        else:
            mean_dist_diff_values.append(np.nan)
            std_dist_diff_values.append(np.nan)

        if e_for_current_sigma:
            mean_e_values.append(np.mean(e_for_current_sigma))
            std_e_values.append(np.std(e_for_current_sigma))
        else:
            mean_e_values.append(np.nan)
            std_e_values.append(np.nan)

        if pif_e_for_current_sigma:
            mean_pif_e_values.append(np.mean(pif_e_for_current_sigma))
            std_pif_e_values.append(np.std(pif_e_for_current_sigma))
        else:
            mean_pif_e_values.append(np.nan)
            std_pif_e_values.append(np.nan)
        
        sigma_values_for_plot.append(float(sigma_val))

    # --- Plotting the Comprehensive Chalice ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting Standard Deviations (the Chalice)
    ax.set_title(r'Comprehensive Chalice: Geometric Stability vs. $\sigma$', color='white', fontsize=18)
    ax.set_xlabel(r'Real Part ($\sigma$)', color='white')
    ax.set_ylabel('Standard Deviation (Normalized)', color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # Normalize all standard deviation values for a clear comparison
    std_dist_diff_norm = np.array(std_dist_diff_values) / np.nanmax(std_dist_diff_values)
    std_e_norm = np.array(std_e_values) / np.nanmax(std_e_values)
    std_pif_e_norm = np.array(std_pif_e_values) / np.nanmax(std_pif_e_values)

    ax.plot(sigma_values_for_plot, std_dist_diff_norm, color='lime', linestyle='-', linewidth=2, label=r'Distance Diff. ($\sigma$)')
    ax.plot(sigma_values_for_plot, std_e_norm, color='cyan', linestyle='--', linewidth=2, label=r'Eccentricity ($e$)')
    ax.plot(sigma_values_for_plot, std_pif_e_norm, color='magenta', linestyle=':', linewidth=2, label=r'$\Pi_f(e)$')

    ax.axvline(0.5, color='gold', linestyle='--', linewidth=1.5, label=r'Critical Line ($\sigma=0.5$)')
    
    ax.legend(facecolor='black', edgecolor='gray', labelcolor='white')
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_chalice.png'))
    plt.show()

if __name__ == "__main__":
    main()
