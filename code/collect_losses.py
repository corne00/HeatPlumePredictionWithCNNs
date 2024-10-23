import numpy as np
import matplotlib.pyplot as plt
import pathlib
import json

results_dir = pathlib.Path("/scratch/e451412/code/results/pkixy_5000_new")
losses = {"MAELoss":[],
    "MSELoss":[],
    "CombiLoss (a=0.25)":[],
    "CombiLoss (a=0.5)":[],
    "CombiLoss (a=0.75)":[],
    "ThresholdedMSE (t=0.02, w=0.1)":[],
    "WeightedMSELoss (e=0.1)":[],
    "RMSELoss":[],
}

for model_dir in results_dir.iterdir():
    if (model_dir/"loss_comparison.json").exists():
        with open(model_dir/"loss_comparison.json", "r") as f:
            loss_comparison = json.load(f)
            for name_loss, value_loss in loss_comparison.items():
                losses[name_loss].append([int(model_dir.name), float(value_loss)])

for name_loss, value_loss in losses.items():
    losses[name_loss] = np.array(value_loss)

plt.figure(figsize=(10,30))
for number, name_loss in enumerate(losses):
    plt.subplot(len(losses),1,number+1)
    plt.plot(losses[name_loss][:,0], np.log(losses[name_loss][:,1]), "x")
    plt.title(name_loss)
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.tight_layout()
    print(f"{name_loss} - min: {np.round(losses[name_loss][:,1].min(),4)}, which trial: {int(losses[name_loss][losses[name_loss][:,1].argmin(),0])}; max: {np.round(losses[name_loss][:,1].max(),4)}, which trial: {int(losses[name_loss][losses[name_loss][:,1].argmax(),0])}")
plt.savefig(results_dir/"loss_log_comparison.png")


plt.figure(figsize=(10,30))
for number, name_loss in enumerate(losses):
    plt.subplot(len(losses),1,number+1)
    plt.plot(losses[name_loss][:,0], losses[name_loss][:,1], "x")
    plt.title(name_loss)
    plt.xlabel("Trial")
    plt.ylabel("Loss")
    plt.tight_layout()
    print(f"{name_loss} - min: {np.round(losses[name_loss][:,1].min(),4)}, which trial: {int(losses[name_loss][losses[name_loss][:,1].argmin(),0])}; max: {np.round(losses[name_loss][:,1].max(),4)}, which trial: {int(losses[name_loss][losses[name_loss][:,1].argmax(),0])}")
plt.savefig(results_dir/"loss_comparison.png")

# to collective csv
top_row = ["Trial"]
for name_loss in losses:
    top_row.append(name_loss)

with open(results_dir/"loss_comparison.csv", "w") as f:
    f.write(",".join(top_row)+"\n")
    for i in range(len(losses["MAELoss"])):
        f.write(str(int(losses["MAELoss"][i,0]))+","+",".join([str(np.round(losses[name_loss][i,1],4)) for name_loss in losses])+"\n")
        