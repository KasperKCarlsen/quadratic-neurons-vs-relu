import subprocess

commands = [
    ["python", "Train_Model.py", "--model", "Baseline", "--ReLU", "2", "--epoch", "100", "--num_blocks", "5", "5", "5"],
    ["python", "Train_Model.py", "--model", "Baseline", "--ReLU", "1", "--epoch", "100", "--num_blocks", "5", "5", "5"],
    ["python", "Train_Model.py", "--model", "Baseline", "--ReLU", "0", "--epoch", "100", "--num_blocks", "5", "5", "5"],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        break  # remove this line if you want to continue on failure