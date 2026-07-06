# config.py
# Central configuration for the v2 (production-oriented) pipeline. Keeping every
# tunable constant in one place is a standard production practice: it means
# retraining with different settings never requires touching model code.

# ---- fleet simulation size (v2 uses a much larger fleet than the v1 MVP) ----
N_AIRCRAFT = 40                              # v1 used 8; v2 simulates a 5x larger fleet
COMPONENTS = ["engine", "landing_gear", "brakes", "bogie"]
MIN_LIFE, MAX_LIFE = 150, 320                 # total lifespan range (cycles) per part
RUL_CAP = 130                                  # standard "clipped RUL" ceiling used for training

# ---- sensor channels ----
# Named and unit-tagged after real Flight Data Recorder / Engine Health Monitoring (EHM)
# parameter conventions (ARINC 717/FAA AC 20-141B define ~88 recorded parameters on
# real aircraft) rather than generic/unitless names - see data_simulator_v2.py for the
# per-component physical meaning of each channel (e.g. temperature_c = EGT for engine,
# brake/bearing temperature for the other 3 components).
SENSOR_COLUMNS = ["vibration_mm_s", "temperature_c", "pressure_psi", "oil_debris_ppm", "rotational_speed_deviation_pct"]

# ---- feature engineering ----
ROLLING_WINDOW = 5                            # trailing window size (cycles) for rolling features
MIN_HEALTHY_ROWS = 10                          # minimum rows used to establish a part's healthy baseline

# ---- Isolation Forest hyperparameters (anomaly detection, scikit-learn) ----
# Now that this runs on scikit-learn's compiled/vectorized implementation rather
# than a pure-Python from-scratch version, we can afford a larger, more standard
# ensemble size without the runtime cost that capped these values in v2's first pass.
IFOREST_N_ESTIMATORS = 200                     # sklearn default is 100; 200 for extra stability
IFOREST_MAX_SAMPLES = 256                       # rows sampled per tree ("auto" in sklearn = min(256, n))
IFOREST_CONTAMINATION = 0.05                    # assumed fraction of anomalous rows (also used for our own
                                                 # validation-tuned threshold, see anomaly_detection_v2.py)
IFOREST_MAX_FEATURES = 1.0                      # fraction of features considered per tree
IFOREST_N_JOBS = -1                             # use all available CPU cores
IFOREST_RANDOM_SEED = 7

# ---- Random Forest hyperparameters (RUL regression, scikit-learn) ----
RF_N_ESTIMATORS = 300                           # sklearn's C-optimized trees make a much larger forest cheap
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_SPLIT = 10
RF_MIN_SAMPLES_LEAF = 4
RF_MAX_FEATURES = "sqrt"                        # classic Random Forest feature-bagging rule of thumb
RF_N_JOBS = -1                                  # use all available CPU cores
RF_RANDOM_SEED = 11

# ---- train / validation / test split (by unit_id, never by row) ----
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SPLIT_SEED = 3

# ---- dashboard thresholds ----
CRITICAL_RUL = 25
WARNING_RUL = 60

# ---- file paths (relative to src_v2/) ----
DATA_DIR = "../data_v2"
OUTPUT_DIR = "../outputs_v2"
ARTIFACT_DIR = "artifacts"                      # trained model files + model card live here
