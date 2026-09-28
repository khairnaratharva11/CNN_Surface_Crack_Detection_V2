import streamlit as st
import pandas as pd
import os
from db.database import Database
from core.config import config
from core.comparison import rank_sessions
import plotly.express as px

st.set_page_config(page_title="Compare & Prioritize", page_icon="📈", layout="wide")
st.title("Compare & Prioritize Sessions")

db = Database(config['database']['path'])
all_sessions = db.get_sessions()

if not all_sessions:
    st.info("No sessions available to compare.")
    st.stop()

session_opts = {}
for s in all_sessions:
    session_opts[f"{s['start_time'][:16]} - {s['site_tag']}"] = s['id']
    
st.markdown("Select two or more sessions to compare severity profiles and prioritize repair actions.")

selected_names = st.multiselect("Select Sessions", list(session_opts.keys()), default=list(session_opts.keys())[:3] if len(session_opts) > 0 else None)

if not selected_names:
    st.info("Please select at least one session.")
    st.stop()
    
selected_ids = [session_opts[n] for n in selected_names]

ranked = rank_sessions(selected_ids, db)

if not ranked:
    st.warning("Could not compute rankings for selected sessions.")
    st.stop()
    
top_session = ranked[0]
st.error(f"**Work on this first:** {top_session['site_tag']} — {top_session['justification']}")

# Build DataFrame for Table
table_data = []
for i, r in enumerate(ranked):
    table_data.append({
        "Rank": i + 1,
        "Site Tag": r['site_tag'],
        "Priority Score": f"{r['priority_score']:.1f}",
        "Critical": r['summary']['counts_by_severity']['Critical'],
        "High": r['summary']['counts_by_severity']['High'],
        "Justification": r['justification'],
        "Session ID": r['session_id']
    })
    
df_table = pd.DataFrame(table_data)
st.dataframe(df_table, use_container_width=True)

# Charts
st.subheader("Severity Breakdown Comparison")

chart_data = []
for r in ranked:
    sid = r['site_tag']
    counts = r['summary']['counts_by_severity']
    for sev in ["Critical", "High", "Moderate", "Low"]:
        chart_data.append({
            "Site": sid,
            "Severity": sev,
            "Count": counts[sev]
        })
        
df_chart = pd.DataFrame(chart_data)
fig = px.bar(df_chart, x="Site", y="Count", color="Severity", barmode="group",
             color_discrete_map={"Critical": "red", "High": "orange", "Moderate": "yellow", "Low": "green"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Export")
if st.button("Export Priority Report to CSV"):
    out_dir = config['output'].get('reports_dir', 'reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multi_site_priority_report.csv")
    df_table.to_csv(out_path, index=False)
    st.success(f"Exported to {out_path}")
