import subprocess

commands = [
    ["python", "Train_Model.py", "--model", "Quad_neuron", "--ReLU", "2", "--epoch", "25", "--num_blocks", "5", "5", "5","--n_trials","10"],
    ["python", "Train_Model.py", "--model", "Quad_neuron", "--ReLU", "1", "--epoch", "25", "--num_blocks", "5", "5", "5","--n_trials","10"],
    ["python", "Train_Model.py", "--model", "Quad_neuron", "--ReLU", "0", "--epoch", "25", "--num_blocks", "5", "5", "5","--n_trials","10"],
]

for i, cmd in enumerate(commands, 1):
    print(f"Running: {' '.join(cmd)}")
    with open(f"logs/run_{i}.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=f)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        break  # remove this line if you want to continue on failure