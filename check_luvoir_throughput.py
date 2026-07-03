import re

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from prysm import coordinates, geometry, propagation

# Load mask arrays from FITS
with fits.open("luvoir_files/TelAp.fits") as hdul:
    TelAp = hdul[0].data.astype(np.float64)
    hdr_telap = hdul[0].header

with fits.open("luvoir_files/Apod.fits") as hdul:
    Apod = hdul[0].data.astype(np.float64)
    asdf_raw = bytes(hdul[1].data[0][0]).decode("utf-8")

with fits.open("luvoir_files/LS.fits") as hdul:
    LS = hdul[0].data.astype(np.float64)
    hdr_ls = hdul[0].header


def _float(key, text):
    return float(re.search(rf"{key}:\s*([\d.]+)", text).group(1))


def _int(key, text):
    return int(re.search(rf"{key}:\s*(\d+)", text).group(1))


fpm_sec = re.search(r"focal_plane_mask:\s*\{([^}]+)\}", asdf_raw).group(1)
img_sec = re.search(r"image:\s*\{([^}]+)\}", asdf_raw).group(1)

fpm_radius_lod = _float("radius", fpm_sec)  # 3.5 λ/D
iwa_lod = _float("iwa", img_sec)  # 3.4 λ/D
owa_lod = _float("owa", img_sec)  # 12.0 λ/D
bw = _float("bandwidth", img_sec)  # 0.10
n_wl = _int("num_wavelengths", img_sec)  # 5

D_circ_m = hdr_telap["D_CIRC"]
D_insc_m = hdr_telap["D_INSC"]
N_pupil = TelAp.shape[0]

D_circ_mm = 1.0
D_insc_mm = D_circ_mm * (D_insc_m / D_circ_m)
dx_pupil = D_circ_mm / N_pupil
efl_mm = 100.0
wl0_um = 0.500
wls_um = wl0_um * np.linspace(1.0 - bw / 2, 1.0 + bw / 2, n_wl)
lod_um = wl0_um * efl_mm / D_insc_mm

q_fpm = 8
fpm_dx_um = lod_um / q_fpm
N_fpm = 256
x_fpm, y_fpm = coordinates.make_xy_grid(N_fpm, dx=fpm_dx_um)
r_fpm = np.hypot(x_fpm, y_fpm)
fpm_radius_um = fpm_radius_lod * lod_um
fpm_disk = geometry.circle(fpm_radius_um, r_fpm).astype(np.float64)

q_img = 8
img_dx_um = lod_um / q_img

# ─────────────────────────────────────────────────────────────────────────────
# Encircled Energy Relative Throughput Calculation
# ─────────────────────────────────────────────────────────────────────────────


def calculate_throughput(x_min=0.0, x_max=owa_lod, n_points=None):
    if n_points is None:
        n_points = int(round((x_max - x_min) * 4)) + 1

    offsets = np.linspace(x_min, x_max, n_points)
    print(
        f"Calculating throughput for {n_points} points from {x_min} to {x_max} λ/D..."
    )

    # Pre-generate pupil grid
    x_p, y_p = coordinates.make_xy_grid(N_pupil, dx=dx_pupil)

    # We will use a sufficiently large image grid for the shifted PSFs.
    # To be safe, we make N_img_tc cover at least max(offsets) + 4 λ/D
    N_img_tc = max(256, int(2 * np.ceil((max(offsets) + 4) * q_img)))
    if N_img_tc % 2 != 0:
        N_img_tc += 1

    x_img, y_img = coordinates.make_xy_grid(N_img_tc, dx=img_dx_um)
    x_img_lod = x_img / lod_um
    y_img_lod = y_img / lod_um

    E_pupil_base = (TelAp * Apod).astype(complex)

    throughputs = []

    for i, x_off in enumerate(offsets):
        psf_coro_off = np.zeros((N_img_tc, N_img_tc), dtype=float)
        psf_tel_off = np.zeros((N_img_tc, N_img_tc), dtype=float)

        # Photometry mask centered at the off-axis position
        r_phot = np.hypot(x_img_lod - x_off, y_img_lod)
        phot_mask = r_phot <= 0.7

        for wl in wls_um:
            # Phase tilt for off-axis source (achromatic angular position)
            phase_tilt = np.exp(
                1j * 2 * np.pi * x_off * x_p / D_insc_mm * (wl0_um / wl)
            )

            # Coronagraphic path with off-axis source
            E_pupil_coro = E_pupil_base * phase_tilt
            E_blocked = propagation.to_fpm_and_back(
                wavefunction=E_pupil_coro.copy(),
                dx=dx_pupil,
                efl=efl_mm,
                wavelength=wl,
                fpm=fpm_disk,
                fpm_dx=fpm_dx_um,
            )
            E_lyot = (E_pupil_coro - E_blocked) * LS

            E_sci_coro = propagation.focus_fixed_sampling(
                wavefunction=E_lyot,
                input_dx=dx_pupil,
                prop_dist=efl_mm,
                wavelength=wl,
                output_dx=img_dx_um,
                output_samples=N_img_tc,
            )
            psf_coro_off += np.abs(E_sci_coro) ** 2

            # Telescope-only path with off-axis source
            E_pupil_tel = TelAp.astype(complex)  # * phase_tilt
            E_sci_tel = propagation.focus_fixed_sampling(
                wavefunction=E_pupil_tel,
                input_dx=dx_pupil,
                prop_dist=efl_mm,
                wavelength=wl,
                output_dx=img_dx_um,
                output_samples=N_img_tc,
            )
            psf_tel_off += np.abs(E_sci_tel) ** 2
            PLOT_INTERMEDIATES = False
            if PLOT_INTERMEDIATES:
                plt.figure()
                plt.subplot(121)
                plt.imshow(np.abs(E_sci_coro) * phot_mask, cmap="gray")
                plt.title(f"Coronagraphic PSF (λ = {wl} um)")
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(np.abs(E_sci_tel) * phot_mask, cmap="gray")
                plt.title(f"Telescope PSF (λ = {wl} um)")
                plt.colorbar()
                plt.show()

        half_max = np.max(psf_coro_off) / 2.0
        phot_mask = psf_coro_off >= half_max

        energy_coro = np.sum(psf_coro_off[phot_mask])
        energy_tel = np.sum(psf_tel_off)

        val = energy_coro / energy_tel if energy_tel > 0 else 0.0
        throughputs.append(val)
        print(f"  x_off = {x_off:5.2f} λ/D  ->  Throughput = {val:6.4f}")

    return offsets, np.array(throughputs)


offsets, throughputs = calculate_throughput()

# Plot the throughput curve
plt.figure(figsize=(8, 5))
plt.plot(offsets, throughputs, "o-", color="purple", lw=2, label="Relative Throughput")
plt.axvline(iwa_lod, color="orange", ls="--", label=f"IWA ({iwa_lod} λ/D)")
plt.axvline(
    fpm_radius_lod, color="red", ls=":", label=f"FPM radius ({fpm_radius_lod} λ/D)"
)
plt.axvline(owa_lod, color="red", ls="--", label=f"OWA ({owa_lod} λ/D)")
plt.xlabel("Off-axis source separation (λ/D)")
plt.ylabel("Encircled Energy Relative Throughput (0.7 λ/D radius)")
plt.title("APLC Encircled Energy Relative Throughput")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("test_throughput.png", dpi=150)
print("Finished!")
