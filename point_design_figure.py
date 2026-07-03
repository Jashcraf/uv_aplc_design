# The regular stuff
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from astropy.io import fits
from pathlib import Path
import time
import numpy as tnp
import sys
sys.path.append('../poi_dependecies/prysm/')
sys.path.append('../poi/')

# The prysm stuff
from prysm.mathops import np, set_backend_to_cupy
from prysm.propagation import focus_fixed_sampling
from prysm.fttools import MatrixDFTExecutor

from poi.masks import ImgSamplingSpec, inner_core_mask, annular_mask, lyot_mask
from poi.processing import azimuthal_average 

colors = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7"
    "#9c755f",
    "#bab0ac",
]

# Construct coronagraph and build throughput plot
IWA = 6
# --- USER INPUT DESIGN PARAMS HERE
USE_GPU = False # Use GPU for the optimization
EPD = 24.4381  # milimeters
EFL = EPD * 40 # milimeters
WVL = 0.350 # microns
IMG_NPIX = (256 + 128) // 1
#IWA = 6
OWA = 20
AZMIN = -65 / 2 # Defines the angular extend of the dark zone
AZMAX = 65 / 2
BANDWIDTH = 10 # percent
NWVLS = 5
OVERSAMPLE = 8 # pix per lam/D
pth_to_aperture = Path.home() / "poi/luvoir_b_pupil_512px.fits"
LS_FRAC = 0.95 # Fraction of the pupil radius to use for the Lyot stop
LS_OBSCURATION_RATIO = 0.00 # Ratio of the Lyot stop obscuration to the pupil radius
core_size = 0.7 # radius in lam/D
# 1e-11 produces good monochromatic designs

THROUGHPUT_LOG_WEIGHT = 13
# ---

if USE_GPU:
    # np switches from numpy to cupy
    set_backend_to_cupy()

tilt_lds = np.arange(0, OWA, .5)

mdft = MatrixDFTExecutor()
mdft.clear()

# Set up the bandpass
half_bw = BANDWIDTH / 2 / 100
band = np.linspace(WVL * (1-half_bw), WVL * (1 + half_bw), NWVLS)
print(band)

# Set the FPM inner working angle and outer working angle to have margin before dark hole
FPM_IWA = IWA
FPM_OWA = OWA
print(f"FPM = {FPM_IWA}-{FPM_OWA}")

# Load the aperture
aperture = np.round(np.array(fits.getdata(pth_to_aperture)))
PUPIL_NPIX = aperture.shape[0]

# Create the focal plane mask
img_dx = WVL * (EFL / EPD) / OVERSAMPLE
pupil_dx = EPD / PUPIL_NPIX

psf = focus_fixed_sampling(wavefunction=aperture,
                           input_dx=pupil_dx,
                           prop_dist=EFL,
                           wavelength=WVL,
                           output_dx=img_dx,
                           output_samples=IMG_NPIX)

lambd = EFL / EPD * WVL
psf_scalar = np.abs(psf)**2
iss = ImgSamplingSpec(IMG_NPIX, lambd / OVERSAMPLE, lambd)
focal_plane_mask = annular_mask(iss, FPM_IWA, FPM_OWA, theta_min=AZMIN, theta_max=AZMAX)
focal_plane_mask += np.fliplr(focal_plane_mask)
dh = annular_mask(iss, IWA, OWA, theta_min=AZMIN, theta_max=AZMAX)
dh = dh + np.fliplr(dh)
ls_mask = lyot_mask(PUPIL_NPIX, pupil_dx=pupil_dx, frac=LS_FRAC, obscuration_ratio=LS_OBSCURATION_RATIO)

# Load the computed mask

mask_path = Path.home() / "Downloads/bw_survey" \
            / "LUVOIRB_512Npup_192Nimg_6IWA_20OWA_65AZ_0LSinner_0.95LSouter_10.0throughput_weight_12.0bw.fits"


newmask = np.array(fits.getdata(mask_path))


def prop_coro(aplc, fpm, ls, wave=WVL, tilt=0, include_fpm=True):

    # Handle passing a float
    if isinstance(wave, float):
        wave = [wave]

    pupil_npix = PUPIL_NPIX


    before_fpm_intensity = 0
    before_ls_intensity = 0
    coro_img_onax_intensity = 0

    for wvl in wave:
        
        # get the tilt phase
        x = np.linspace(-0.5, 0.5, pupil_npix)
        tilt_phase = np.exp(1j * 2 * np.pi * x * tilt * (WVL / wvl))

        before_fpm = focus_fixed_sampling(
                    wavefunction= aplc * tilt_phase,
                    input_dx=pupil_dx,
                    prop_dist = EFL,
                    wavelength= wvl,
                    output_dx= img_dx,
                    output_samples=(IMG_NPIX, IMG_NPIX),
                    shift=(0, 0),
                    method='mdft')

        if include_fpm:
            before_fpm *= fpm

        before_ls = focus_fixed_sampling(
                    wavefunction=before_fpm,
                    input_dx=img_dx,
                    prop_dist = EFL,
                    wavelength= wvl,
                    output_dx= pupil_dx,
                    output_samples=(pupil_npix, pupil_npix),
                    shift=(0, 0),
                    method='mdft')

        coro_img_onax = focus_fixed_sampling(
                    wavefunction=before_ls * ls,
                    input_dx=pupil_dx,
                    prop_dist = EFL,
                    wavelength= wvl,
                    output_dx= img_dx,
                    output_samples=(IMG_NPIX, IMG_NPIX),
                    shift=(0, 0),
                    method='mdft')

        # accumulate intensities
        before_fpm_intensity += np.abs(before_fpm)**2
        before_ls_intensity += np.abs(before_ls)**2
        coro_img_onax_intensity += np.abs(coro_img_onax)**2

    return before_fpm_intensity, before_ls_intensity, coro_img_onax_intensity

fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
fx, fy = np.meshgrid(fx, fx)
rx = np.sqrt(fx**2 + fy**2)
mask = np.zeros_like(rx, dtype=int)
mask[rx < core_size] = 1

# Get the reference PSF
before, ls, coro = prop_coro(aperture, np.ones_like(focal_plane_mask), ls_mask, tilt=0, wave=band)
throughput_ref = np.sum(before[mask==1])
contrast_ref = coro.max()

# Get the contrast profile
before, ls, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=0, wave=band)
profile, bins = azimuthal_average(coro / contrast_ref, angle_range=[AZMIN, AZMAX])

# 14 mins uh oh
from matplotlib.colors import LogNorm
from tqdm import tqdm
throughput = []
throughput_07 = []

for i, ld in tqdm(enumerate(tilt_lds)):

    before, ls, coro = prop_coro(newmask, focal_plane_mask, ls_mask, tilt=ld, wave=band)
    before_I = np.sum(before)
    coro_I = coro
    throughput.append(np.sum(coro_I) / np.sum(NWVLS * aperture))

    # get value in 0.7 L/D (recall 1/OS is pixelscale in L/D)
    fx = np.linspace(-IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX / (2 * OVERSAMPLE), IMG_NPIX)
    fx, fy = np.meshgrid(fx, fx)
    fx += ld
    rx = np.sqrt(fx**2 + fy**2)
    mask = np.zeros_like(rx, dtype=int)
    mask[rx < core_size] = 1

    value_in_aperture = np.sum(coro_I[mask==1])
    throughput_07.append(value_in_aperture / throughput_ref)


lamd_ticks = bins / OVERSAMPLE 

import hwostyle
hwostyle.use("light")

plt.figure(figsize=[10,4.2])
plt.subplot(131)
plt.imshow(newmask, cmap="gray")
plt.xticks([],[])
plt.yticks([],[])
plt.subplot(132)
plt.title("Azimuthal Average Profile")
plt.ylabel("Normalized Intensity")
plt.plot(lamd_ticks, profile, color=colors[0], label="Azimuthal Average")
plt.xlabel('Angular Separation, '+r'$\lambda / D$')
plt.axvline(x=6, color=colors[1], label="IWA")
plt.ylim(1e-12, 1e-9)
plt.xlim(0, 20)
plt.plot(lamd_ticks, np.ones_like(lamd_ticks) * 1e-10, label=r"$10^{-10}$", linestyle="dashed", color="k")
plt.legend()
plt.yscale("log")
plt.subplot(133)
plt.title("Field PSF Core Throughput")

if np.__name__ == "cupy":
    # plt.plot(tilt_lds.get(), np.array(throughput).get(), linestyle='dashed', color=color, label=f"{IWA}"+r"$\lambda/D$")
    plt.plot(tilt_lds.get(), np.array(throughput_07).get(), linestyle='solid', color=color, label=f"{IWA}"+r"$\lambda/D$")
else:
    plt.plot(tilt_lds, np.array(throughput_07), linestyle='solid', color=colors[2])

plt.xlabel('Angular Separation, '+r'$\lambda / D$')
plt.ylabel('Throughput')
plt.ylim(0, 1.0)
plt.xlim(0, OWA)
plt.subplots_adjust()
plt.show()

