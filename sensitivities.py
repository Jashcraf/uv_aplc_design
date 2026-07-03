from pathlib import Path
import matplotlib.pyplot as plt
import hwostyle
import numpy as np
import pandas as pd
hwostyle.use("light")
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

sensitivities_pth = Path.home() / "Downloads/Results_LS_BW_Survey_RJuanola"

keys = ["Bandwidth", "LS_Edges", "LS_Segments"]
types = ["Global", "Segment"]
bandwidths = np.arange(10, 32, 2)
ls_radii = np.arange(80, 100, 5)
zindex = np.arange(1, 8, 1)
zlabels = [
    "Piston",
    "Tip/Tilt",
    "Focus",
    "Astigmatism",
    "Coma",
    "Trefoil",
    "Spherical",
]

sensitivities = {}

# Scan .csvs for changes in aberration sensitivity
for t in types:
    for key in keys:

        if key == "Bandwidth":
            
            indvar = bandwidths
            date = "05222026"
            xlabel = "Bandwidth [%]"
            title = t

        else:

            indvar = ls_radii
            date = "05182026"
            xlabel = "Fractional LS Radius"
            title = "Segment-to-Segment"
        
        piston = []
        tiptilt = []
        focus = []
        astigmatism = []
        coma = []
        trefoil = []
        spherical = []

        for i in indvar:
            pth = sensitivities_pth / f"{key}_{i}_01_{date}_{t}_Zernike_Sensitivities.csv"
            data = pd.read_csv(pth)
            loaded_sensitivity = data['Sensitivity_at_4e-11_ppt/pm']

            # 2ith element is full dark zone
            # 2i + 1th element is ring at 6.5 L/D
            # Parse IWA
            piston.append(loaded_sensitivity[1])
            tiptilt.append(loaded_sensitivity[3])
            focus.append(loaded_sensitivity[5])
            astigmatism.append(loaded_sensitivity[7])
            coma.append(loaded_sensitivity[9])
            trefoil.append(loaded_sensitivity[11])
            spherical.append(loaded_sensitivity[13])
        
        sensitivities = [
            piston,
            tiptilt,
            focus,
            astigmatism,
            coma,
            trefoil,
            spherical
        ]
        plt.figure()
        plt.title(f"{key} Low-order Aberration Sensitivity - {title}")
        for sensitivity, zlabel, color in zip(sensitivities, zlabels, colors):
            plt.plot(indvar, sensitivity, label=zlabel, marker="o", color=color)
        plt.legend()
        plt.ylabel(f"Sensitivity at 4e-11 Contrast [ppt/pm]")
        plt.xlabel(xlabel)
        plt.show()
