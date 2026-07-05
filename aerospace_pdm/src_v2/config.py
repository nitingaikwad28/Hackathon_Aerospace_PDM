# config.py
# Central configuration for the v2 (production-oriented) pipeline. Keeping every
# tunable constant in one place is a standard production practice: it means
# retraining with different settings never requires touching model code.

# ---- fleet simulation size (v2 uses a much larger fleet than the v1 MVP) ----
N_AIRCRAFT = 40                              # v1 used 8; v2 simulates a 5x larger fleet
COMPONENTS = ["engine", "landing_gear", "brakes", "bogie"]
MIN_LIFE, MAX_LIFE = 150, 320                 # total lifespan range (cycles) per part
RUL_CAP = 130                                  # standard "clipped RUL" ceiling used for training

# ---- sensor channels (v2 adds 2 more channels than v1's 3) ----
SENSOR_COLUMNS = ["vibration", "temperature", "pressure", "oil_debris", "rpm_deviation"]

# ---- feature engineering ----
ROLLING_WINDOW = 5                            # trailing window size (cycles) for rolling features
MIN_HEALTHY_ROWS = 10                          # minimum rows used to establish a part's healthy baseline

# ---- Isolation Forest hyperparameters (anomaly detection) ----
IFOREST_N_TREES = 100                          # number of isolation trees per component model
IFOREST_SUBSAMPLE_SIZE = 256                   # rows sampled (without replacement) to build each tree
IFOREST_MAX_DEPTH = None                        # None -> auto = ceil(log2(subsample_size))
IFOREST_CONTAMINATION = 0.05                    # assumed fraction of anomalous rows, used to pick the score cutoff
IFOREST_RANDOM_SEED = 7

# ---- Random Forest hyperparameters (RUL regression) ----
RF_N_TREES = 25                                  # tuned down from sklearn-typical 100+ to keep from-scratch
RF_MAX_DEPTH = 6                                 # numpy training fast enough for a live sandbox demo
RF_MIN_SAMPLES_SPLIT = 20
RF_MIN_SAMPLES_LEAF = 10
RF_MAX_FEATURES_FRACTION = 0.6                   # fraction of features considered at each split (feature bagging)
RF_HISTOGRAM_BINS = 8                            # candidate split thresholds per feature (histogram-based, fast)
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
