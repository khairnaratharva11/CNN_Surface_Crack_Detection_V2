import streamlit as st
from db.database import Database
from core.config import config
from core.reports import generate_pdf_report, export_csv
import os

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")
st.title("Generate Reports")

db = Database(config['database']['path'])
sessions = db.get_sessions()

if not sessions:
    st.info("No sessions available to report on.")
    st.stop()

session_opts = {}
for s in sessions:
    session_opts[f"{s['start_time'][:16]} - {s['site_tag']}"] = s['id']
    
selected_session = st.selectbox("Select Session for Report", list(session_opts.keys()))
session_id = session_opts[selected_session]

col1, col2 = st.columns(2)

with col1:
    st.subheader("PDF Inspection Report")
    st.write("Generates a formatted PDF containing session details, summary counts, and top critical cracks with images.")
    if st.button("Generate PDF"):
        with st.spinner("Generating PDF..."):
            pdf_path = generate_pdf_report(session_id, config['database']['path'])
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f.read(), file_name=os.path.basename(pdf_path), mime="application/pdf")
            else:
                st.error("Failed to generate PDF.")

with col2:
    st.subheader("CSV Export")
    st.write("Exports the raw database records for the selected session to a CSV file.")
    if st.button("Generate CSV"):
        with st.spinner("Generating CSV..."):
            csv_path = export_csv(session_id, config['database']['path'])
            if csv_path and os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button("Download CSV", f.read(), file_name=os.path.basename(csv_path), mime="text/csv")
            else:
                st.error("Failed to generate CSV.")
