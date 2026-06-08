import subprocess
import os
import sys

os.makedirs("Run_number", exist_ok=True)

commands = [
    ["python", "Train_Model.py", "--model", "Quad_neuron", "--ReLU", "2", "--epoch", "25", "--num_blocks", "5", "5", "5","--n_trials","10"],
    ["python", "Train_Model.py", "--model", "Quad_neuron", "--ReLU", "1", "--epoch", "25", "--num_blocks", "5", "5", "5","--n_trials","10"],
    ["python", "Train_Model.py", "--model", "Quad_neuron", "--ReLU", "0", "--epoch", "25", "--num_blocks", "5", "5", "5","--n_trials","10"],
]

for i, cmd in enumerate(commands, 1):
    print(f"Running: {' '.join(cmd)}")
    with open(f"Run_number/run_{i}.txt", "w") as f:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
        process.wait()
    if process.returncode != 0:
        print(f"Command failed with code {process.returncode}")
        break  # remove this line if you want to continue on failure