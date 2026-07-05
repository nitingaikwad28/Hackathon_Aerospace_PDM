# dashboard_v2.py
# Same idea as v1's dashboard.py (fleet-level, prioritized health table + HTML view),
# adapted for the v2 models: predicted_RUL now comes from the Random Forest, and
# is_anomaly comes from the per-component Isolation Forest + tuned threshold.

import pandas as pd

from config import CRITICAL_RUL, WARNING_RUL


def classify_health(row):
    if row["predicted_RUL"] <= CRITICAL_RUL:
        return "Critical"
    elif row["predicted_RUL"] <= WARNING_RUL or row["is_anomaly"] == 1:
        return "Warning"
    else:
        return "Healthy"


def build_dashboard(scored_df):
    # scored_df must already have predicted_RUL, anomaly_score, is_anomaly columns
    latest = scored_df.sort_values("cycle").groupby("unit_id").tail(1).copy()
    latest["health_status"] = latest.apply(classify_health, axis=1)
    latest["priority_rank"] = latest["predicted_RUL"] - (latest["is_anomaly"] * 30)
    dashboard_df = latest.sort_values("priority_rank")

    columns_to_show = ["aircraft_id", "unit_id", "component", "cycle",
                        "predicted_RUL", "anomaly_score", "is_anomaly", "health_status"]
    return dashboard_df[columns_to_show].reset_index(drop=True)


def _metrics_panel_html(metrics_summary):
    if not metrics_summary:
        return ""
    anomaly = metrics_summary.get("anomaly_detection", {})
    rul_overall = metrics_summary.get("rul_overall", {})

    def stat(label, value):
        return (f"<div style='display:inline-block; margin:0 22px 10px 0; text-align:center'>"
                f"<div style='font-size:22px; font-weight:bold; color:#1a3c6e'>{value}</div>"
                f"<div style='font-size:11px; color:#666'>{label}</div></div>")

    html = "<div style='background:#eef3f9; border:1px solid #cdd9e5; border-radius:8px; padding:14px 18px; margin-bottom:16px'>"
    html += "<h3 style='margin:0 0 10px 0; color:#1a3c6e'>Model Performance Benchmarks (v2 - Random Forest + Isolation Forest)</h3>"
    html += stat("RUL MAE (cycles)", rul_overall.get("MAE", "n/a"))
    html += stat("RUL RMSE (cycles)", rul_overall.get("RMSE", "n/a"))
    html += stat("Anomaly recall", f"{anomaly.get('recall', 0)*100:.0f}%")
    html += stat("Early-life false alarm rate", f"{anomaly.get('early_life_false_alarm_rate', 0)*100:.1f}%")
    html += stat("Mean detection lead time", f"{anomaly.get('mean_detection_lead_time_cycles', 'n/a')} cyc")
    html += ("<div style='font-size:10.5px; color:#777; margin-top:6px'>Metrics computed on the held-out TEST "
             "split (units never used for training). See model_metrics_v2.json for the full breakdown.</div>")
    html += "</div>"
    return html


def save_dashboard_html(dashboard_df, path, metrics_summary=None):
    color_map = {"Critical": "#ffcccc", "Warning": "#fff3cd", "Healthy": "#d4edda"}
    rows_html = ""
    for _, row in dashboard_df.iterrows():
        color = color_map.get(row["health_status"], "white")
        rows_html += (
            f"<tr style='background-color:{color}'>"
            f"<td>{row['aircraft_id']}</td><td>{row['unit_id']}</td><td>{row['component']}</td>"
            f"<td>{row['cycle']}</td><td>{row['predicted_RUL']:.1f}</td>"
            f"<td>{row['anomaly_score']:.3f}</td><td>{row['is_anomaly']}</td>"
            f"<td><b>{row['health_status']}</b></td></tr>"
        )

    metrics_html = _metrics_panel_html(metrics_summary)

    html = f"""
    <html><head><title>Fleet Predictive Maintenance Dashboard (v2)</title></head>
    <body style="font-family:Arial">
    <h2>Aerospace Predictive Maintenance - Fleet Dashboard (v2: Random Forest + Isolation Forest)</h2>
    {metrics_html}
    <table border="1" cellpadding="6" cellspacing="0">
    <tr><th>Aircraft</th><th>Unit</th><th>Component</th><th>Cycle</th>
    <th>Predicted RUL</th><th>Anomaly Score</th><th>Anomaly?</th><th>Status</th></tr>
    {rows_html}
    </table></body></html>
    """
    with open(path, "w") as f:
        f.write(html)
