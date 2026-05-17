# ARM32 Instruction Scheduling Project

Welcome to our project for the INFO-H410 course (Artificial Intelligence at ULB).

For this project, we tried to solve an interesting problem: how do we schedule ARM32 instructions for masked cryptography without leaking secrets through side channels? 

We basically had to write a tool that reorders instructions so that we don't accidentally put two different shares too close to each other in the processor pipeline (which would cause a Hamming-distance leakage).

## What we did

As specified by the instructions of the project, we tested different approaches to see what works best:

1. **Bayesian Network**: We built a probabilistic model that calculates the risk of a leakage happening based on the distance between instructions. If the cumulative risk is too high, we throw in a NOP.
2. **CSP with OR-Tools**: We wanted to see what the mathematically perfect schedule looks like, so we modeled the whole thing as a constraint satisfaction problem. It guarantees a strict distance `k` between shares.
3. **Deep Q-Learning**: We trained an RL agent (a DQN) to play the scheduling game. We gave it rewards and penalties based on how many cycles it used and if it violated the security rules.

## Code Structure

We split the code into a few parts to keep things organized:

- `arm_scheduler/core/`: This is where we put the base structures. We have `instruction.py` to represent ARM32 instructions and `pipeline.py` to handle the RAW dependencies and our security logic.
- `arm_scheduler/solvers/`: Here are the three main algorithms we wrote. You'll find `bayesian.py`, `csp.py`, and `mdp.py` here.
- `arm_scheduler/evaluation/`: We put our benchmarking code here so we could compare our solvers fairly.
- `experiments/`: This folder has the scripts we used to run everything and generate data. `run_all.py` is the main one we used for the results.
- `report/`: Contains the LaTeX files for our final academic report.

## How to run our code

If you want to test what we did, we included a few ways to run it.

If you are on Windows and have an NVIDIA GPU, you can just run our batch script:
```powershell
./run_gpu_windows.bat
```

We also set up Docker so it's easy to run without messing up your local python environment. You can build it and run our benchmark like this:
```bash
docker build -t arm-scheduler .
docker run --rm --gpus all -v "${PWD}/experiments:/app/experiments" arm-scheduler python3 experiments/run_all.py

# To run the generalist DQN training and evaluation:
docker run --rm --gpus all -v "${PWD}/experiments:/app/experiments" arm-scheduler python3 experiments/train_generalist.py
```

Or if you just want to run it locally, install the dependencies and run the script:
```bash
pip install -r requirements.txt
pip install -e .
python experiments/run_all.py --methods bayesian csp mdp --k 3
```
