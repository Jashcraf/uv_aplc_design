from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize_scalar

try:
    from prysm.coordinates import make_xy_grid, cart_to_polar
    from prysm.geometry import spider, circle, offset_circle
    from prysm.segmented import CompositeHexagonalAperture
except ImportError as e:
    print(f"Error {e} \n while importing prysm, remember to pip install prysm!")

"""LUVOIR-B Parameters
"""
EQUIV_LS_RADIUS = 0.6
PLOT_PUPIL = True
SAVE_PUPIL = False
CIRCUMSCRIBED_DIAMETER = 7.994 # * (1 - PERCENT_DEGRADATION/100) 
FLAT_TO_FLAT = 0.955 #* (1 - PERCENT_DEGRADATION/100)
GAP_SIZE = 0.006
N_RINGS = 4
exclude = [
    37,
    41,
    45,
    49,
    53,
    57
]


if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        N_SAMPLES = float(sys.argv[1])
    else:
        N_SAMPLES = 1024

    circular_ls_radius = EQUIV_LS_RADIUS * CIRCUMSCRIBED_DIAMETER
    dx = CIRCUMSCRIBED_DIAMETER / N_SAMPLES

    x, y = make_xy_grid(N_SAMPLES, diameter=CIRCUMSCRIBED_DIAMETER)
    x += dx / 2
    y += dx / 2
    r, t = cart_to_polar(x, y)
    circular = circle(circular_ls_radius / 2, r).astype(np.float32)

    def loss(pct):
        """minimization of throughput difference

        Parameters
        ----------
        x : float
            percent degradation of the aperture edge

        Returns
        -------
        float:
            MSE of throughput between circular (fixed) and segmented (variable)
        """

        # Abs means only decreases are allowed
        pct = np.abs(pct)
        gap_size = GAP_SIZE + FLAT_TO_FLAT * pct/100
        flat_to_flat = FLAT_TO_FLAT * (1 - pct/100)

        # Draw new aperture
        cha = CompositeHexagonalAperture(x, y, N_RINGS, flat_to_flat, gap_size, exclude=exclude)
        aperture = cha.amp.astype(np.float32)
        squared_err = (np.sum(circular) - np.sum(aperture))**2

        return squared_err
     
    results = minimize_scalar(loss, bounds=[0, 50])
    print(results)
    PERCENT_DEGRADATION = np.abs(results.x)
    gap_size = GAP_SIZE + FLAT_TO_FLAT * PERCENT_DEGRADATION/100
    flat_to_flat = FLAT_TO_FLAT * (1 - PERCENT_DEGRADATION/100)
    cha = CompositeHexagonalAperture(x, y, N_RINGS, flat_to_flat, gap_size, exclude=exclude)

    if PLOT_PUPIL:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title(f"LUVOIR-B Lyot Stop with Segment Index - {PERCENT_DEGRADATION:.2f}% Degradation")
        ax.imshow(cha.amp, origin='lower', cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
        for center, id_ in zip(cha.all_centers, cha.segment_ids):
            plt.text(*center, id_)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()
    
    if SAVE_PUPIL:
        hdu = fits.PrimaryHDU(cha.amp.astype(np.float64))
        hdu.writeto(f"luvoir_b_pupil_{N_SAMPLES}_shift_px_py.fits")
