# Aerospace Predictive Maintenance - Hackathon MVP

An AI-powered predictive maintenance demo for aerospace fleets: it simulates sensor
data, detects early signs of component degradation, predicts Remaining Useful Life
(RUL), benchmarks both models with maintenance-relevant metrics, and produces a
prioritized fleet dashboard that feeds into an MRO (Maintenance, Repair & Overhaul)
system.

Built with only **numpy** and **pandas** - no internet connection or GPU needed to run.

## How the 5 requirements map to this code

| # | Requirement | File(s) |
|---|---|---|
| 1 | Working anomaly detection prototype on a representative sensor dataset | `data_simulator.py` + `anomaly_detection.py` |
| 2 | RUL estimation model with accuracy benchmarks | `rul_model.py` + `evaluation.py` |
| 3 | Fleet-level maintenance dashboard with prioritized alerts | `dashboard.py` |
| 4 | Evaluation metrics incl. detection lead time & false alarm rate | `evaluation.py` |
| 5 | Integration roadmap for MRO/asset-management connectivity | `mro_integration.py` + roadmap below |

`main.py` runs all of the above end-to-end in one command.

## How it works

1. **Simulate telemetry** (`data_simulator.py`) - creates realistic sensor readings
   (vibration, temperature, pressure) for a fleet of 8 aircraft x 4 components each.
   Each part has its own random age and lifespan, so the fleet looks like a real one:
   mostly healthy, a few aging, one or two near failure. Ground-truth columns
   (`RUL`, `true_anomaly`, `life_length`) are kept so the models can be objectively
   benchmarked.
2. **Detect anomalies** (`anomaly_detection.py`) - learns each part's own healthy
   sensor baseline (mean + spread, from at least its first 10 cycles) and flags any
   later reading that is statistically far from that baseline (z-score > 4).
3. **Predict RUL** (`rul_model.py`) - one linear regression per component type
   (closed-form, via `numpy.linalg.lstsq`), mapping current sensor readings + cycle
   number to remaining cycles of life.
4. **Benchmark both models** (`evaluation.py`) - see the Model Evaluation section below.
5. **Build the dashboard** (`dashboard.py`) - combines RUL + anomaly status for every
   part's latest reading into one prioritized table (Healthy / Warning / Critical),
   with a model-performance summary panel, saved as `outputs/dashboard.html`.
6. **Push to MRO** (`mro_integration.py`) - turns every Warning/Critical row into a
   structured work order and "sends" it to a mock MRO system (`outputs/work_orders.json`).

## Running the demo

```bash
pip install -r requirements.txt
cd src
python main.py
```

This produces:
- `data/fleet_telemetry.csv` - raw simulated sensor data
- `outputs/dashboard.html` - color-coded fleet dashboard + model performance panel
- `outputs/fleet_status.csv` - same dashboard data as a spreadsheet
- `outputs/work_orders.json` - generated maintenance work orders
- `outputs/model_metrics.json` - full benchmark report (RUL accuracy + anomaly detection quality)

Each source file can also be run on its own (e.g. `python evaluation.py`) to test
that piece in isolation.

## Model evaluation metrics

From a representative run (`outputs/model_metrics.json`, fleet of 32 simulated parts):

| Metric | Value | What it means |
|---|---|---|
| RUL MAE (overall) | ~13.5 cycles | Average miss on predicted remaining life, held-out parts |
| RUL MAE by component | 4-6 cycles (engine/brakes/bogie), ~21 (landing gear) | Landing gear is the hardest component to predict - see Limitations |
| Anomaly recall | 100% | Every injected fault in the test data was caught |
| Early-life false alarm rate | 0.0% | Zero false flags during the first 20% of a part's life (should be healthy) |
| Whole-life false alarm rate | ~22% | Higher because many flags here are genuine early degradation warnings, not detector errors (see caveat below) |
| Mean detection lead time | ~150 cycles | Average advance warning before a part's true end-of-life, once first flagged |

**Caveat on false alarm rate:** the simulator only labels sudden injected spikes
(e.g. a bird strike or sensor glitch) as `true_anomaly = 1`. Gradual wear-driven
drift is NOT labeled as a true anomaly, even though flagging it is exactly what an
early-warning system should do. That is why we report both numbers: the early-life
rate isolates genuine false alarms (sensor noise on healthy parts), while the
whole-life rate also counts (correct) early degradation catches as "false" relative
to the spike-only label. Both are computed in `evaluation.py::evaluate_anomaly_detection`.

## Integration roadmap: MRO & asset-management connectivity

`mro_integration.py` is currently a mock adapter (writes `work_orders.json` instead
of calling a real system). The intended rollout path to a live MRO/asset-management
system:

1. **Phase 1 - Data contract validation (this MVP).** File/JSON-based mock adapter
   proves out the work-order schema (component, predicted RUL, anomaly flag,
   priority, recommended action) before any live system is involved.
2. **Phase 2 - Pilot API integration.** Replace `send_to_mro_system` with a REST
   adapter calling a sandbox/test instance of the target system (e.g. SAP PM, IBM
   Maximo, Trax, AMOS), using that system's native work-order/notification API.
   Endpoint and credentials become config, not code.
3. **Phase 3 - Secure production connectivity.** Add OAuth2/API-key authentication,
   retry + idempotency handling (so a network blip never double-creates a work
   order), and a field-mapping/crosswalk layer between our schema and the target
   system's data model (functional locations, asset numbers, etc.), plus audit
   logging of every transmitted work order.
4. **Phase 4 - Bi-directional sync.** Pull completed work-order status and
   maintenance history back from the MRO system, so the RUL model can be retrained
   on real outcomes and each part's life counter resets correctly after maintenance.
5. **Phase 5 - Real-time, fleet-scale rollout.** Move from batch/cron pipeline runs
   to event-driven streaming (e.g. a message queue) so a live anomaly reaches the
   MRO system within seconds, plus role-based views for planners vs. technicians.

## Notes on the "simple by design" choices

- No deep learning / external ML libraries: keeps setup to two `pip install`s and
  keeps the logic auditable for a live demo Q&A.
- One linear model *per component type* because engine/landing-gear/brakes/bogie
  sensors sit on very different scales (e.g. engine temperature ~550 vs landing
  gear ~60); separate small models fit each much better than one mixed model.
- All thresholds (`Z_SCORE_LIMIT`, `CRITICAL_RUL`, `WARNING_RUL`, `HEALTHY_FRACTION`)
  are plain constants at the top of their files - easy to retune live during a demo.

## Limitations

- Trained and evaluated entirely on synthetic data; thresholds are hand-picked, not
  tuned against real failure records.
- Landing gear RUL accuracy (~21 cycle MAE) is noticeably worse than the other three
  components in this run - likely too few training examples for that component type
  in a fleet this small (8 aircraft). A larger simulated (or real) fleet would help.
- Before production use: retrain/validate on real sensor and maintenance-outcome
  data, and connect `mro_integration.py` to a real system per the roadmap above.
