# state_definitions.py
import numpy as np

def generate_state_features(state):
    """
    Generate base behavioural metrics based on behavioural state.
    """

    if state == 0:  # Stable
        logins = np.random.poisson(25)
        delay = np.random.exponential(0.5)
        attendance = np.random.normal(90, 5)
        sentiment = np.random.normal(0.6, 0.1)

    elif state == 1:  # Strain
        logins = np.random.poisson(18)
        delay = np.random.exponential(2)
        attendance = np.random.normal(80, 7)
        sentiment = np.random.normal(0.1, 0.2)

    elif state == 2:  # Withdrawal
        logins = np.random.poisson(10)
        delay = np.random.exponential(4)
        attendance = np.random.normal(65, 10)
        sentiment = np.random.normal(-0.3, 0.2)

    else:  # Burnout
        logins = np.random.poisson(4)
        delay = np.random.exponential(7)
        attendance = np.random.normal(45, 15)
        sentiment = np.random.normal(-0.7, 0.2)

    night_ratio = np.clip(np.random.normal(0.3 + state*0.1, 0.1), 0, 1)

    return logins, delay, attendance, sentiment, night_ratio