import subprocess
import os

os.makedirs("Run_number", exist_ok=True)

commands = [
    ["python", "Train_Model.py", "--model", "Baseline", "--ReLU", "2", "--epoch", "50", "--num_blocks", "5", "5", "5", "--compile", "none","--n_trials","50","--classes","0","1","2","3","4","5","6","7","8"],
    ["python", "Train_Model.py", "--model", "Baseline", "--ReLU", "1", "--epoch", "50", "--num_blocks", "5", "5", "5", "--compile", "none","--n_trials","50","--classes","0","1","2","3","4","5","6","7","8"],
    ["python", "Train_Model.py", "--model", "Baseline", "--ReLU", "0", "--epoch", "50", "--num_blocks", "5", "5", "5", "--compile", "none","--n_trials","50","--classes","0","1","2","3","4","5","6","7","8"],
]

for i, cmd in enumerate(commands, 1):
    print(f"Running: {' '.join(cmd)}")
    with open(f"Run_number/run_{i}.txt", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=f)
