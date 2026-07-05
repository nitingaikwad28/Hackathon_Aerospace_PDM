# data_simulator.py
# This file creates FAKE (simulated) sensor data for an aircraft fleet.
# We pretend we are reading real sensors, so we can demo the system without real aircraft data.
# Each part in the fleet is simulated up to its OWN current point in life, so the fleet
# looks realistic: most parts healthy, some aging, a few close to failure - just like real life.

import numpy as np                      # numpy helps us make numbers and math easier
import pandas as pd                     # pandas helps us store data in table format (like Excel)

# ---- SETTINGS (easy to change later) ----
COMPONENTS = ["engine", "landing_gear", "brakes", "bogie"]   # 4 aircraft parts we are monitoring
N_AIRCRAFT = 8                          # how many aircraft are in our fleet
RUL_CAP = 130                           # we cap "Remaining Useful Life" at 130 cycles (industry practice)
MIN_LIFE, MAX_LIFE = 150, 300           # each part fails somewhere between these many cycles

# Each component has its own normal ("healthy") sensor readings and noise level
SENSOR_BASELINE = {
    "engine":        {"vibration": 0.30, "temperature": 550, "pressure": 40},
    "landing_gear":  {"vibration": 0.15, "temperature": 60,  "pressure": 3000},
    "brakes":        {"vibration": 0.20, "temperature": 120, "pressure": 1500},
    "bogie":         {"vibration": 0.25, "temperature": 80,  "pressure": 200},
}
SENSOR_NOISE = {
    "engine":        {"vibration": 0.02, "temperature": 5,  "pressure": 1},
    "landing_gear":  {"vibration": 0.01, "temperature": 3,  "pressure": 50},
    "brakes":        {"vibration": 0.015,"temperature": 8,  "pressure": 30},
    "bogie":         {"vibration": 0.02, "temperature": 4,  "pressure": 10},
}


def make_one_unit_data(rng, aircraft_id, component, unit_id):
    # rng = random number generator (so every run can be reproducible)
    # aircraft_id = e.g. "AC-001"; component = e.g. "engine"; unit_id = unique id for this part

    total_life = int(rng.integers(MIN_LIFE, MAX_LIFE))   # total cycles this part will last before failing
    # "current_age" = how far into its life this part is RIGHT NOW (like a real fleet snapshot in time).
    # We bias it towards earlier cycles so most parts in the fleet are healthy, a few are close to failure.
    current_age = int(rng.triangular(1, total_life * 0.3, total_life))
    current_age = min(current_age, total_life)            # never let it go past the part's total life

    cycles = np.arange(1, current_age + 1)                # cycle numbers we will simulate: 1, 2, ... current_age
    wear = (cycles / total_life) ** 2                      # wear grows slowly then speeds up near end-of-life

    rows = []                                              # we will collect one row per cycle here
    base = SENSOR_BASELINE[component]                      # normal readings for this component
    noise = SENSOR_NOISE[component]                        # noise level for this component

    # decide if THIS unit will get a random sudden anomaly (like a bird strike or sensor glitch)
    has_anomaly = rng.random() < 0.2                       # 20% chance this unit gets a random spike
    anomaly_cycle = rng.integers(5, current_age) if (has_anomaly and current_age > 10) else -1  # pick spike cycle

    for i, cyc in enumerate(cycles):                       # loop through every simulated cycle for this unit
        # vibration rises as the part wears out (more shaking = more damage)
        vibration = base["vibration"] + wear[i] * base["vibration"] * 1.5 + rng.normal(0, noise["vibration"])
        # temperature rises as the part wears out (more friction/heat = more damage)
        temperature = base["temperature"] + wear[i] * base["temperature"] * 0.25 + rng.normal(0, noise["temperature"])
        # pressure falls as the part wears out (leaks/seal wear = less pressure)
        pressure = base["pressure"] - wear[i] * base["pressure"] * 0.2 + rng.normal(0, noise["pressure"])

        is_anomaly = 0                                     # default: this cycle is NOT an anomaly
        if cyc == anomaly_cycle:                           # if this is the "surprise spike" cycle
            vibration += base["vibration"] * 3               # sudden spike in vibration
            is_anomaly = 1                                   # mark this cycle as a true anomaly

        rul = max(0, min(RUL_CAP, total_life - cyc))         # remaining life = cycles left, capped at RUL_CAP

        rows.append([aircraft_id, unit_id, component, cyc, vibration, temperature, pressure,
                     rul, is_anomaly, total_life])

    # turn our list of rows into a pandas table with named columns
    # (life_length is the part's TRUE total lifespan, kept uncapped - used only for evaluation,
    #  e.g. measuring detection lead time. The RUL column above is what the model trains on.)
    return pd.DataFrame(rows, columns=[
        "aircraft_id", "unit_id", "component", "cycle",
        "vibration", "temperature", "pressure", "RUL", "true_anomaly", "life_length"
    ])


def simulate_fleet(seed=42):
    # This builds the FULL fleet: every aircraft x every component, each at its own point in life
    rng = np.random.default_rng(seed)                      # fixed seed = same fake data every run (reproducible demo)
    all_data = []                                            # list to collect every unit's table

    unit_counter = 0                                         # counter to make unique unit ids
    for a in range(1, N_AIRCRAFT + 1):                       # loop over each aircraft
        aircraft_id = f"AC-{a:03d}"                          # e.g. "AC-001"
        for component in COMPONENTS:                          # loop over each component type
            unit_counter += 1                                # increase the unique id counter
            unit_id = f"{aircraft_id}-{component}-{unit_counter}"   # unique name for this part
            unit_df = make_one_unit_data(rng, aircraft_id, component, unit_id)  # simulate this one part
            all_data.append(unit_df)                          # add it to our collection

    fleet_df = pd.concat(all_data, ignore_index=True)         # stack all the small tables into one big table
    return fleet_df                                            # give back the full fleet table


if __name__ == "__main__":
    # this block only runs if we execute this file directly (not when imported)
    df = simulate_fleet()                                     # build the fake fleet data
    df.to_csv("../data/fleet_telemetry.csv", index=False)      # save it as a CSV file for later use
    print(f"Created {df['unit_id'].nunique()} units and {len(df)} total sensor readings.")
    print(df.groupby("unit_id")["cycle"].max().describe())      # sanity check: spread of "current age" across fleet
