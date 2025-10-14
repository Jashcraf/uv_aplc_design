from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    from prysm.coordinates import make_xy_grid, cart_to_polar
    from prysm.geometry import spider, circle, offset_circle
    from prysm.segmented import CompositeHexagonalAperture
except ImportError as e:
    print(f"Error {e} \n while importing prysm, remember to pip install prysm!")

"""LUVOIR-B Parameters
"""
PLOT_PUPIL = True
CIRCUMSCRIBED_DIAMETER = 7.994
FLAT_TO_FLAT = 0.955
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
        N_SAMPLES = sys.argv[1]
    else:
        N_SAMPLES = 1024

    x, y = make_xy_grid(N_SAMPLES, diameter=CIRCUMSCRIBED_DIAMETER)
    cha = CompositeHexagonalAperture(x, y, N_RINGS, FLAT_TO_FLAT, GAP_SIZE, exclude=exclude)

    if PLOT_PUPIL:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.title("LUVOIR-B Pupil with Segment Index")
        ax.imshow(cha.amp, origin='lower', cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()])
        for center, id_ in zip(cha.all_centers, cha.segment_ids):
            plt.text(*center, id_)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()

    hdu = fits.PrimaryHDU(cha.amp.astype(np.float64))
    hdu.writeto(f"luvoir_b_pupil_{N_SAMPLES}px.fits")
