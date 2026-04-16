from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift(ref_df, curr_df):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)
    report.save_html("drift_report.html")