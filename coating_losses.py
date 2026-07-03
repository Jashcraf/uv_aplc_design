import numpy as np
import matplotlib.pyplot as plt
import hwostyle

hwostyle.use("light")

reflectances = [0.8, 0.85, 0.9, 0.95, 0.99]
n_optics = np.logspace(0, 2, 100)

tableau_10 = {
#"Cyan": "#17becf",
#"Olive": "#bcbd22",
#"Gray": "#7f7f7f",
#"Pink": "#e377c2",
#"Brown": "#8c564b",
"Purple": "#9467bd",
"Red": "#d62728",
"Green": "#2ca02c",
"Orange": "#ff7f0e",
"Blue": "#1f77b4"
}

hwo_like = [
    "M1",
    "M2",
    "M3",
    "BSC",
    "FSM",
    "OAP1",
    "OAP2",
    "DM1",
    "DM2",
    "OAP3",
    "OAP4",
    "APODIZER",
    "OAP5",
    "FPM",
    "OAP6",
    "LS",
    "OAP7",
    "FIELDSTOP",
    "OAP8",
    "OAP9",
    "OAP10"
]

plt.figure()

for r, color in zip(reflectances, tableau_10):
    refl = r ** n_optics
    plt.plot(n_optics, refl, label=f"R={r:.2f}", color=color)

plt.xlabel("Number of Optics")
plt.ylabel("Fractional Transmission")
plt.xscale("log")
plt.ylim([0, 1.])
plt.xlim([1, 100.])
plt.axvline(len(hwo_like), ymin=0, ymax=1, linestyle="dashed", label="EAC1-like",color="k")
plt.axvline(47, ymin=0, ymax=1, linestyle="dotted", label="GMT-like", color="k")
plt.legend()
plt.show()
