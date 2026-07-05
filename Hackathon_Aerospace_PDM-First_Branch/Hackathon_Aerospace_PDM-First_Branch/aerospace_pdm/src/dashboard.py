# dashboard.py
# This file builds the "Maintenance Action Dashboard" - a single table that shows,
# for every part in the fleet, its current health status, so planners can decide
# what to fix first. It also saves a simple HTML page so it can be viewed in a browser.

import pandas as pd                          # for tables

CRITICAL_RUL = 25                             # if predicted RUL is below this -> Critical (fix very soon)
WARNING_RUL = 60                              # if predicted RUL is below this -> Warning (plan maintenance)


def classify_health(row):
    # decide the health "traffic light" for one part, based on its RUL prediction and anomaly flag.
    # Critical is reserved for parts that are TRULY close to failure (low predicted RUL).
    # A live sensor anomaly on its own is downgraded to "Warning" - worth investigating, not yet an emergency.
    if row["predicted_RUL"] <= CRITICAL_RUL:
        return "Critical"                      # very little life left -> needs attention now
    elif row["predicted_RUL"] <= WARNING_RUL or row["is_anomaly"] == 1:
        return "Warning"                       # either aging or showing unusual readings -> plan maintenance soon
    else:
        return "Healthy"                       # nothing to do yet


def build_dashboard(scored_df, weights, predict_rul_fn, feature_columns):
    # scored_df = fleet data that already has anomaly_score / is_anomaly columns attached
    latest = scored_df.sort_values("cycle").groupby("unit_id").tail(1).copy()  # keep only the LAST cycle per part
    latest["predicted_RUL"] = predict_rul_fn(weights, latest)                   # predict RUL for that latest cycle

    latest["health_status"] = latest.apply(classify_health, axis=1)             # assign Healthy/Warning/Critical

    # priority: lower RUL = fix sooner = should be at the TOP of the list.
    # we also nudge anomalies higher up the list, since a live fault deserves a closer look sooner.
    latest["priority_rank"] = latest["predicted_RUL"] - (latest["is_anomaly"] * 30)
    dashboard_df = latest.sort_values("priority_rank")                          # most urgent first

    columns_to_show = ["aircraft_id", "unit_id", "component", "cycle",
                        "predicted_RUL", "anomaly_score", "is_anomaly", "health_status"]
    return dashboard_df[columns_to_show].reset_index(drop=True)                 # tidy final table


def _metrics_panel_html(metrics_summary):
    # Builds a small "Model Performance" summary panel shown above the fleet table,
    # so the dashboard also communicates how trustworthy the underlying models are.
    if not metrics_summary:                                       # nothing to show if no metrics were passed in
        return ""
    anomaly = metrics_summary.get("anomaly_detection", {})          # anomaly detector benchmark numbers
    rul_overall = metrics_summary.get("rul_overall", {})             # overall RUL MAE/RMSE

    def stat(label, value):
        return (f"<div style='display:inline-block; margin:0 22px 10px 0; text-align:center'>"
                f"<div style='font-size:22px; font-weight:bold; color:#1a3c6e'>{value}</div>"
                f"<div style='font-size:11px; color:#666'>{label}</div></div>")

    html = "<div style='background:#eef3f9; border:1px solid #cdd9e5; border-radius:8px; padding:14px 18px; margin-bottom:16px'>"
    html += "<h3 style='margin:0 0 10px 0; color:#1a3c6e'>Model Performance Benchmarks</h3>"
    html += stat("RUL MAE (cycles)", rul_overall.get("MAE", "n/a"))
    html += stat("RUL RMSE (cycles)", rul_overall.get("RMSE", "n/a"))
    html += stat("Anomaly recall", f"{anomaly.get('recall', 0)*100:.0f}%")
    html += stat("Early-life false alarm rate", f"{anomaly.get('early_life_false_alarm_rate', 0)*100:.1f}%")
    html += stat("Mean detection lead time", f"{anomaly.get('mean_detection_lead_time_cycles', 'n/a')} cyc")
    html += ("<div style='font-size:10.5px; color:#777; margin-top:6px'>Early-life false alarm rate = wrong "
             "flags during the first 20% of a part's life (should be healthy). The whole-life false alarm "
             "rate is higher because many later flags are genuine early degradation warnings rather than "
             "detector mistakes - see model_metrics.json / documentation for the full breakdown.</div>")
    html += "</div>"
    return html


def save_dashboard_html(dashboard_df, path, metrics_summary=None):
    # turn the summary table into a very simple, readable HTML page (no internet/JS libraries needed)
    color_map = {"Critical": "#ffcccc", "Warning": "#fff3cd", "Healthy": "#d4edda"}  # row colors per status

    rows_html = ""                                                              # will hold all table rows as text
    for _, row in dashboard_df.iterrows():                                      # go through each part's summary row
        color = color_map.get(row["health_status"], "white")                    # pick row color based on status
        rows_html += (
            f"<tr style='background-color:{color}'>"
            f"<td>{row['aircraft_id']}</td><td>{row['unit_id']}</td><td>{row['component']}</td>"
            f"<td>{row['cycle']}</td><td>{row['predicted_RUL']:.1f}</td>"
            f"<td>{row['anomaly_score']:.2f}</td><td>{row['is_anomaly']}</td>"
            f"<td><b>{row['health_status']}</b></td></tr>"
        )

    metrics_html = _metrics_panel_html(metrics_summary)                          # build the metrics panel (if any)

    html = f"""
    <html><head><title>Fleet Predictive Maintenance Dashboard</title></head>
    <body style="font-family:Arial">
    <h2>Aerospace Predictive Maintenance - Fleet Dashboard</h2>
    {metrics_html}
    <table border="1" cellpadding="6" cellspacing="0">
    <tr><th>Aircraft</th><th>Unit</th><th>Component</th><th>Cycle</th>
    <th>Predicted RUL</th><th>Anomaly Score</th><th>Anomaly?</th><th>Status</th></tr>
    {rows_html}
    </table></body></html>
    """
    with open(path, "w") as f:                    # open the file for writing
        f.write(html)                              # write our generated HTML into it


if __name__ == "__main__":
    # quick manual test: build the full pipeline and print/save the dashboard
    from data_simulator import simulate_fleet
    from anomaly_detection import detect_anomalies
    from rul_model import split_train_test, train_linear_model, predict_rul, FEATURE_COLUMNS, evaluate, TARGET_COLUMN
    from evaluation import evaluate_anomaly_detection, evaluate_rul_by_component

    fleet = simulate_fleet()                        # 1. make fake sensor data
    scored = detect_anomalies(fleet)                # 2. flag anomalies
    train_df, test_df = split_train_test(fleet)     # 3. split for training/testing
    weights = train_linear_model(train_df)          # 4. train RUL model
    test_predictions = predict_rul(weights, test_df)

    all_metrics = {
        "rul_overall": evaluate(test_df[TARGET_COLUMN].values, test_predictions),
        "rul_by_component": evaluate_rul_by_component(test_df, test_predictions),
        "anomaly_detection": evaluate_anomaly_detection(scored),
    }

    dash = build_dashboard(scored, weights, predict_rul, FEATURE_COLUMNS)  # 5. build final summary table
    print(dash.to_string(index=False))               # print it nicely to the console
    save_dashboard_html(dash, "../outputs/dashboard.html", metrics_summary=all_metrics)  # save with metrics panel
    print("\nSaved dashboard.html")
