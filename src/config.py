# config.py

RANDOM_SEED = 42

# Dataset size
N_STUDENTS = 6000
N_WEEKS = 12

# Behavioural states
STATES = {
    0: "Stable",
    1: "Strain",
    2: "Withdrawal",
    3: "Burnout"
}

# State transition probabilities (Markov-style)
TRANSITION_MATRIX = {
    0: [0.75, 0.20, 0.04, 0.01],
    1: [0.10, 0.65, 0.20, 0.05],
    2: [0.05, 0.15, 0.60, 0.20],
    3: [0.02, 0.08, 0.20, 0.70]
}