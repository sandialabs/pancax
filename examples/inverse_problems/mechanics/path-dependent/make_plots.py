from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import re

actual_prop = 1.0
history_file = "history_pd.csv"

df = pandas.read_csv(history_file)
df.columns = df.columns.str.strip()

print(df)

plt.figure(1)
plt.plot(df["epoch"], df["shear_modulus"], color="blue", label="PINN")
plt.hlines([1.], xmin=0, xmax=np.max(df["epoch"]), color="black", linestyle="--", label="Actual")
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.ylabel("Shear Modulus [MPa]")
plt.savefig(os.path.join(Path(__file__).parent, "property_evolution.png"))

plt.figure(2)
plt.plot(df["epoch"], np.abs(df["shear_modulus"] - actual_prop) / np.abs(actual_prop), color="blue", label="PINN")
plt.hlines([0.01], xmin=0, xmax=np.max(df["epoch"]), color="black", linestyle="--", label="1 Percent Error")
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.xscale("log")
plt.ylabel("Relative Error")
plt.ylim((10e-4, 10e1))
plt.yscale("log")
plt.savefig(os.path.join(Path(__file__).parent, "property_relative_error.png"))


log_file = "pancax.log"
# float_pattern = r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
float_pattern = r'Array\(([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?), dtype=float64\)'

losses = []
energies = []
field_losses = []
global_losses = []
residuals = []


with open(log_file, "r") as f:
    for line in f.readlines():
        if "Array" in line:
            floats_in_line = re.findall(float_pattern, line)
            floats_in_line = list(map(float, floats_in_line))
            losses.append(floats_in_line[0])
            energies.append(floats_in_line[1])
            field_losses.append(floats_in_line[2])
            global_losses.append(floats_in_line[3])
            residuals.append(floats_in_line[4])


losses = np.array(losses)
energies = np.array(energies)
field_losses = np.array(field_losses)
global_losses = np.array(global_losses)
residuals = np.array(residuals)

print(losses)

# normalize
losses = losses / losses[0]
field_losses = field_losses / field_losses[0]
global_losses = global_losses / global_losses[0]
residuals = residuals / residuals[0]

plt.figure(3)
plt.plot(losses, label="Total Loss")
plt.plot(field_losses, label="Full Field Data Error")
plt.plot(global_losses, label="Global Data Error")
plt.plot(residuals, label="Residual")
plt.legend(loc="best")
plt.xlabel("Epoch")
plt.xscale("log")
plt.ylabel("Normalized Losses")
plt.yscale("log")
plt.savefig(os.path.join(Path(__file__).parent, "losses.png"))

plt.figure(4)
plt.plot(energies)
plt.xlabel("Epoch")
plt.ylabel("Algorithmic Energy")
plt.savefig(os.path.join(Path(__file__).parent, "algorithmic_energy.png"))
