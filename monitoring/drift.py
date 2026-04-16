import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift(ref_df, curr_df, output_path="artifacts/drift_report.html"):
    os.makedirs("artifacts", exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)
    report.save_html(output_path)

    return output_path