import os
import pandas as pd
from datetime import datetime

def generate_session_report(detections_log, config):
    reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', config['paths']['reports_dir']))
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate CSV
    df = pd.DataFrame(detections_log)
    csv_path = os.path.join(reports_dir, f'session_report_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    # Generate HTML
    html_path = os.path.join(reports_dir, f'session_report_{timestamp}.html')
    html_content = f"""
    <html>
    <head><title>Crack Detection Session Report</title></head>
    <body>
        <h1>Surface Crack Detection Report</h1>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <h2>Summary</h2>
        <p>Total Detections: {len(df)}</p>
        <h3>Details</h3>
        {df.to_html()}
    </body>
    </html>
    """
    with open(html_path, 'w') as f:
        f.write(html_content)
        
    print(f"Session report saved to {csv_path} and {html_path}")
