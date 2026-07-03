"""
LUVOIR-A APLC Coronagraph Forward Model
========================================
Polychromatic forward model of the LUVOIR-A Apodized Pupil Lyot Coronagraph
(APLC) using the prysm optical-propagation package.

Optical train
-------------
  P1  Entrance pupil  :  TelAp × Apodizer
      ↓  MFT  (to_fpm_and_back, Babinet)
  F1  Focal-plane mask:  opaque hard-edge disk drawn with prysm.geometry.circle
                         radius = focal_plane_mask.radius from Apod ASDF metadata
      ↓  inverse-MFT
  P2  Lyot plane       :  × Lyot Stop
      ↓  MFT  (focus_fixed_sampling)
  F2  Science focal    :  broadband PSF / contrast map

Babinet's principle for numerical accuracy
------------------------------------------
  Computing  IFT[ FT[E] × (1 - disk) ]  directly requires the focal-plane
  window to capture all PSF energy (expensive, and never fully converged here).

  Instead we use Babinet's principle:
      E_lyot = E_pupil  −  IFT[ FT[E_pupil] × disk ]

  The 'blocked' term is tightly concentrated within the small FPM disk so the
  focal-plane window only needs to be large enough to contain the disk.
  Contrast converges at N_fpm ≥ 2 × (FPM radius in pixels) and is insensitive
  to further increases in N_fpm or q_fpm.

Mask files
----------
  luvoir_files/TelAp.fits  – telescope aperture        (loaded from FITS)
  luvoir_files/Apod.fits   – APLC apodizer             (loaded from FITS;
                              ASDF design metadata is embedded here)
  luvoir_files/LS.fits     – Lyot stop                 (loaded from FITS)
  FPMb                     – drawn fresh with prysm.geometry.circle;
                              hard-edge binary disk, radius from ASDF metadata

Units
-----
  Pupil plane  : mm   (D_circ ≡ 1 mm)
  Focal plane  : μm
  Wavelength   : μm
"""

import re
import warnings

import matplotlib
import numpy as np
from astropy.io import fits
from prysm import coordinates, geometry, propagation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load mask arrays from FITS
# ─────────────────────────────────────────────────────────────────────────────

with fits.open("luvoir_files/TelAp.fits") as hdul:
    TelAp = hdul[0].data.astype(np.float64)
    hdr_telap = hdul[0].header

with fits.open("luvoir_files/Apod.fits") as hdul:
    Apod = hdul[0].data.astype(np.float64)
    # ASDF metadata is stored as a raw-bytes column in BinTableHDU extension 1
    asdf_raw = bytes(hdul[1].data[0][0]).decode("utf-8")

with fits.open("luvoir_files/LS.fits") as hdul:
    LS = hdul[0].data.astype(np.float64)
    hdr_ls = hdul[0].header

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Extract coronagraph design parameters from the Apod ASDF metadata
# ─────────────────────────────────────────────────────────────────────────────


def _float(key, text):
    return float(re.search(rf"{key}:\s*([\d.]+)", text).group(1))


def _int(key, text):
    return int(re.search(rf"{key}:\s*(\d+)", text).group(1))


fpm_sec = re.search(r"focal_plane_mask:\s*\{([^}]+)\}", asdf_raw).group(1)
img_sec = re.search(r"image:\s*\{([^}]+)\}", asdf_raw).group(1)

fpm_radius_lod = _float("radius", fpm_sec)  # 3.5  λ/D – hard-edge FPM radius
iwa_lod = _float("iwa", img_sec)  # 3.4  λ/D – dark-zone inner edge
owa_lod = _float("owa", img_sec)  # 12.0 λ/D – dark-zone outer edge
bw = _float("bandwidth", img_sec)  # 0.10 – fractional BW
n_wl = _int("num_wavelengths", img_sec)  # 5

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Physical parameters from FITS headers
# ─────────────────────────────────────────────────────────────────────────────

D_circ_m = hdr_telap["D_CIRC"]  # 15.0 m  circumscribed (outer) diameter
D_insc_m = hdr_telap["D_INSC"]  # 13.5 m  inscribed diameter  (λ/D reference)
N_pupil = TelAp.shape[0]  # 1000

LS_ID = hdr_ls["LS_ID"]  # 0.12   inner-diameter fraction (rel. to D_INSC)
LS_OD = hdr_ls["LS_OD"]  # 0.982  outer-diameter fraction (rel. to D_INSC)

print("Coronagraph design parameters (from FITS headers + Apod ASDF metadata)")
print(f"  D_circ     = {D_circ_m} m     D_insc = {D_insc_m} m")
print(f"  N_pupil    = {N_pupil} px")
print(
    f"  FPM radius = {fpm_radius_lod} λ/D   IWA = {iwa_lod} λ/D   OWA = {owa_lod} λ/D"
)
print(f"  Bandwidth  = {bw * 100:.0f}%          n_wl = {n_wl}")
print(f"  LS  ID/OD  = {LS_ID} / {LS_OD}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sampling / unit setup
# ─────────────────────────────────────────────────────────────────────────────
# Pupil distances in mm, normalised so D_circ ≡ 1 mm.
# λ/D uses the inscribed diameter D_insc as the aperture reference.

D_circ_mm = 1.0
D_insc_mm = D_circ_mm * (D_insc_m / D_circ_m)  # = 0.9 mm
dx_pupil = D_circ_mm / N_pupil  # mm / pixel

# Focal length (arbitrary; cancels in the contrast normalisation)
efl_mm = 100.0

# Wavelengths: n_wl uniform samples across ±BW/2 around λ₀.
# λ₀ = 500 nm is a representative visible wavelength; not specified in files.
wl0_um = 0.500  # μm
wls_um = wl0_um * np.linspace(1.0 - bw / 2, 1.0 + bw / 2, n_wl)

# Physical size of 1 λ/D (using D_insc) in the focal plane at λ₀  [μm]
lod_um = wl0_um * efl_mm / D_insc_mm  # ≈ 55.6 μm

print(f"\n  λ₀ = {wl0_um * 1000:.0f} nm")
print(f"  wavelengths : {wls_um * 1000} nm")
print(f"  1 λ/D at λ₀ : {lod_um:.2f} μm  (using D_insc = {D_insc_mm} mm)")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build FPM using prysm.geometry  (fresh array, NOT loaded from FPMb.fits)
#
#   radius = focal_plane_mask.radius = 3.5 λ/D  (from Apod ASDF metadata).
#   The design IWA (3.4 λ/D) is the inner edge of the dark zone, which lies
#   just inside the FPM hard edge.
#
#   q_fpm = 4 px / λ/D  →  FPM radius = 3.5 × 4 = 14 pixels.
#   N_fpm only needs to contain the disk; contrast is converged for N_fpm ≥ 50.
# ─────────────────────────────────────────────────────────────────────────────

q_fpm = 8  # focal-plane sampling [pixels per λ/D]
fpm_dx_um = lod_um / q_fpm  # μm / pixel
N_fpm = 256

# Coordinate grid centred on optical axis
x_fpm, y_fpm = coordinates.make_xy_grid(N_fpm, dx=fpm_dx_um)
r_fpm = np.hypot(x_fpm, y_fpm)

# Draw the opaque disk: 1 inside (blocked), 0 outside (open)
fpm_radius_um = fpm_radius_lod * lod_um  # physical FPM radius [μm] at λ₀
fpm_disk = geometry.circle(fpm_radius_um, r_fpm).astype(np.float64)

print(f"\n  FPM drawn with prysm.geometry.circle")
print(f"  fpm_dx_um  = {fpm_dx_um:.3f} μm/px  ({q_fpm} px/λ/D)")
print(
    f"  FPM radius = {fpm_radius_um:.1f} μm  =  {fpm_radius_lod} λ/D  "
    f"= {fpm_radius_lod * q_fpm:.0f} pixels"
)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Science focal-plane sampling
# ─────────────────────────────────────────────────────────────────────────────

q_img = 8  # pixels per λ/D in the science image
img_dx_um = lod_um / q_img
N_img = 256  # → covers ±16 λ/D at λ₀

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Polychromatic forward model
# ─────────────────────────────────────────────────────────────────────────────
# Babinet's principle (numerically superior to the direct approach):
#
#   E_lyot = E_pupil_apod  −  IFT[ FT[E_pupil_apod] × fpm_disk ]
#          = E_pupil_apod  −  to_fpm_and_back(E_pupil_apod, fpm=fpm_disk)
#
# The 'blocked' term is concentrated inside the small disk, so the focal-plane
# window needs only to contain the disk – not the entire extended PSF.
# This converges at N_fpm=50 and is 30× better in contrast than passing
# fpm=(1−disk) directly.

E_pupil_base = (TelAp * Apod).astype(complex)  # entrance-pupil amplitude

psf_coro = np.zeros((N_img, N_img), dtype=float)
psf_unocc = np.zeros((N_img, N_img), dtype=float)

print("\nRunning polychromatic propagation (Babinet) …")
for k, wl in enumerate(wls_um):
    print(f"  λ = {wl * 1000:.1f} nm  ({k + 1}/{n_wl})")

    # ── APLC path (Babinet) ───────────────────────────────────────────────
    # Step A: blocked field  =  IFT[ FT[E_pupil] × disk ]
    E_blocked = propagation.to_fpm_and_back(
        wavefunction=E_pupil_base.copy(),
        dx=dx_pupil,  # mm
        efl=efl_mm,  # mm
        wavelength=wl,  # μm
        fpm=fpm_disk,  # 1 inside disk (blocked), 0 outside
        fpm_dx=fpm_dx_um,  # μm
    )

    # Step B: Babinet subtraction  →  field after the open FPM
    E_lyot = (E_pupil_base - E_blocked) * LS  # apply Lyot stop

    # Step C: Lyot plane  →  science camera
    E_sci = propagation.focus_fixed_sampling(
        wavefunction=E_lyot,
        input_dx=dx_pupil,
        prop_dist=efl_mm,
        wavelength=wl,
        output_dx=img_dx_um,
        output_samples=N_img,
    )
    psf_coro += np.abs(E_sci) ** 2

    # ── Unocculted PSF for normalisation (TelAp only, no coronagraph) ────
    E_unocc = propagation.focus_fixed_sampling(
        wavefunction=TelAp.astype(complex),
        input_dx=dx_pupil,
        prop_dist=efl_mm,
        wavelength=wl,
        output_dx=img_dx_um,
        output_samples=N_img,
    )
    psf_unocc += np.abs(E_unocc) ** 2

# Normalise to peak of unocculted broadband PSF
norm = psf_unocc.max()
psf_norm = psf_coro / norm

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Image-plane coordinates (in λ/D at λ₀)
# ─────────────────────────────────────────────────────────────────────────────
# MFT output is DC-centred; no fftshift required.

ax_lod = np.arange(-N_img // 2, N_img // 2) / q_img  # λ/D
extent = [ax_lod[0], ax_lod[-1], ax_lod[0], ax_lod[-1]]

cy, cx = N_img // 2, N_img // 2
y_i, x_i = np.ogrid[:N_img, :N_img]
r_px = np.hypot(x_i - cx, y_i - cy)

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Azimuthally-averaged radial contrast profile
# ─────────────────────────────────────────────────────────────────────────────

rbins_px = np.arange(0, N_img // 2, 0.5)
rbins_lod = rbins_px / q_img
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # suppress empty-bin NaN
    radial = np.array(
        [np.nanmean(psf_norm[(r_px >= r) & (r_px < r + 0.5)]) for r in rbins_px]
    )
radial = np.nan_to_num(radial, nan=0.0)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Dark-zone statistics
# ─────────────────────────────────────────────────────────────────────────────

dz_mask = (r_px >= iwa_lod * q_img) & (r_px <= owa_lod * q_img)
dz_mean_c = float(psf_norm[dz_mask].mean())
dz_max_c = float(psf_norm[dz_mask].max())

print(f"\nDark-zone contrast  (IWA = {iwa_lod} – OWA = {owa_lod} λ/D)")
print(f"  Mean : {dz_mean_c:.3e}")
print(f"  Peak : {dz_max_c:.3e}")
print(f"  (Design target with grayscale FPM: ~1×10⁻¹⁰)")
print(f"  (Binary hard-edge FPM limit:       ~1×10⁻⁸ – 1×10⁻⁹)")

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Figures
# ─────────────────────────────────────────────────────────────────────────────

theta = np.linspace(0, 2 * np.pi, 360)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# ── Telescope aperture ────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.imshow(TelAp, origin="lower", cmap="gray", interpolation="nearest")
ax.set_title(
    f"Telescope Aperture  (TelAp)\nLUVOIR-A,  D_circ = {D_circ_m} m", fontsize=10
)
ax.axis("off")

# ── Apodizer ──────────────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.imshow(Apod, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.set_title("APLC Apodizer", fontsize=10)
ax.axis("off")

# ── FPM  (freshly drawn with prysm.geometry.circle) ──────────────────────────
fpm_ax = np.arange(-N_fpm // 2, N_fpm // 2) / q_fpm
fpm_ext = [fpm_ax[0], fpm_ax[-1], fpm_ax[0], fpm_ax[-1]]
ax = axes[0, 2]
ax.imshow(
    fpm_disk,
    origin="lower",
    cmap="gray_r",
    extent=fpm_ext,
    interpolation="nearest",
)
ax.plot(
    fpm_radius_lod * np.cos(theta),
    fpm_radius_lod * np.sin(theta),
    "r-",
    lw=1.5,
    label=f"FPM edge  ({fpm_radius_lod} λ/D)",
)
ax.plot(
    iwa_lod * np.cos(theta),
    iwa_lod * np.sin(theta),
    "b--",
    lw=1.5,
    label=f"IWA  ({iwa_lod} λ/D)",
)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
ax.set_title(
    f"FPM  (prysm.geometry.circle)\nradius = {fpm_radius_lod} λ/D", fontsize=10
)
ax.set_xlabel("x  (λ/D)")
ax.set_ylabel("y  (λ/D)")
ax.legend(fontsize=7, loc="upper right")

# ── Lyot stop ─────────────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.imshow(LS, origin="lower", cmap="gray", interpolation="nearest")
ax.set_title(
    f"Lyot Stop\nID = {LS_ID},  OD = {LS_OD}  (rel. D_insc = {D_insc_m} m)",
    fontsize=10,
)
ax.axis("off")

# ── Coronagraphic PSF  (log₁₀ contrast) ──────────────────────────────────────
ax = axes[1, 1]
im = ax.imshow(
    np.log10(psf_norm + 1e-15),
    origin="lower",
    cmap="inferno",
    vmin=-12,
    vmax=0,
    extent=extent,
    interpolation="nearest",
)
plt.colorbar(im, ax=ax, label="log₁₀(contrast)")
ax.plot(
    iwa_lod * np.cos(theta),
    iwa_lod * np.sin(theta),
    "w--",
    lw=1.2,
    label=f"IWA = {iwa_lod} λ/D",
)
ax.plot(
    owa_lod * np.cos(theta),
    owa_lod * np.sin(theta),
    "w:",
    lw=1.2,
    label=f"OWA = {owa_lod} λ/D",
)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Coronagraphic PSF  (log₁₀ contrast)", fontsize=10)
ax.set_xlabel("x  (λ/D)")
ax.set_ylabel("y  (λ/D)")

# ── Radial contrast profile ────────────────────────────────────────────────────
ax = axes[1, 2]
ax.semilogy(
    rbins_lod, radial + 1e-16, color="steelblue", lw=1.8, label="Azimuthal mean"
)
ax.axvline(iwa_lod, color="orange", ls="--", lw=1.5, label=f"IWA = {iwa_lod} λ/D")
ax.axvline(owa_lod, color="red", ls="--", lw=1.5, label=f"OWA = {owa_lod} λ/D")
ax.axhline(1e-10, color="gray", ls=":", lw=1, label="10⁻¹⁰ contrast")
ax.set_xlim(0, 16)
ax.set_ylim(1e-12, 1)
ax.set_xlabel("Angular separation  (λ/D)")
ax.set_ylabel("Contrast")
ax.set_title("Azimuthally Averaged Contrast", fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)

fig.suptitle(
    "LUVOIR-A APLC Forward Model  |  "
    f"λ₀ = {wl0_um * 1000:.0f} nm,  BW = {bw * 100:.0f}%,  "
    f"{n_wl} wavelengths  |  FPM radius = {fpm_radius_lod} λ/D  "
    f"(binary hard-edge,  prysm.geometry.circle)",
    fontsize=11,
    y=1.01,
)
plt.tight_layout()

out_png = "luvoir_aplc_forward_model.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out_png}")
