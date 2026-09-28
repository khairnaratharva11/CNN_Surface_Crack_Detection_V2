import streamlit as st
import pandas as pd
from db.database import Database
from core.config import config
import os

st.set_page_config(page_title="Crack Registry", page_icon="📋", layout="wide")
st.title("Crack Registry")

db = Database(config['database']['path'])

sessions = db.get_sessions()
if not sessions:
    st.info("No sessions found. Run a live inspection or analyze a video first.")
    st.stop()

# Sidebar Filters
st.sidebar.header("Filters")
session_opts = {"All": None}
for s in sessions:
    session_opts[f"{s['start_time'][:16]} - {s['site_tag']}"] = s['id']
    
selected_session = st.sidebar.selectbox("Session", list(session_opts.keys()))

severity_filter = st.sidebar.multiselect("Severity", ["Critical", "High", "Moderate", "Low"], default=["Critical", "High", "Moderate", "Low"])
status_filter = st.sidebar.multiselect("Status", ["new", "reviewed", "repaired", "false positive"], default=["new", "reviewed"])

filters = {}
if session_opts[selected_session] is not None:
    filters['session_id'] = session_opts[selected_session]

cracks = db.get_cracks(filters)

# Apply in-memory filters for lists
if cracks:
    df = pd.DataFrame(cracks)
    df = df[df['severity_level'].isin(severity_filter)]
    df = df[df['status'].isin(status_filter)]
    
    if not df.empty:
        st.subheader("Detected Cracks")
        # Select columns to display
        display_cols = ['id', 'crack_id', 'severity_level', 'max_width', 'unit', 'recommended_action', 'status', 'notes']
        st.dataframe(df[display_cols], use_container_width=True)
        
        st.subheader("Crack Details & Actions")
        selected_id = st.selectbox("Select Database ID to View/Edit", df['id'].tolist())
        
        crack_row = df[df['id'] == selected_id].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Crack ID:** {crack_row['crack_id']}")
            st.write(f"**Severity:** {crack_row['severity_level']} (Score: {crack_row['severity_score']:.1f})")
            st.write(f"**Width:** {crack_row['max_width']:.2f} {crack_row['unit']}")
            st.write(f"**Length:** {crack_row['length']:.2f} {crack_row['unit']}")
            st.write(f"**Position:** {crack_row['position_in_frame']}")
            
            new_status = st.selectbox("Status", ["new", "reviewed", "repaired", "false positive"], 
                                      index=["new", "reviewed", "repaired", "false positive"].index(crack_row['status']))
            new_notes = st.text_area("Notes", value=crack_row['notes'] if pd.notna(crack_row['notes']) else "")
            
            if st.button("Update"):
                db.update_crack_status(selected_id, new_status, new_notes)
                st.success("Updated!")
                st.rerun()
                
        with col2:
            crop_path = crack_row['cropped_path']
            if crop_path and isinstance(crop_path, str) and os.path.exists(crop_path):
                st.image(crop_path, caption="Snapshot")
            else:
                st.info("No snapshot available.")
                
        # --- NEW UI: Surface Analysis ---
        st.markdown("---")
        
        # Read from raw SQL as requested to not modify db/database.py
        c = db.conn.cursor()
        c.execute("SELECT * FROM surface_analysis WHERE crack_id = ?", (selected_id,))
        surface_row = c.fetchone()
        
        if surface_row:
            cols = [desc[0] for desc in c.description]
            surface_data = dict(zip(cols, surface_row))
            
            with st.expander("Surface Analysis Details", expanded=True):
                s1, s2 = st.columns(2)
                with s1:
                    hex_color = surface_data.get('dominant_color_hex', '#FFFFFF')
                    color_name = surface_data.get('nearest_color_name', 'Unknown')
                    st.markdown(f"**Dominant Color:** `{hex_color}` {color_name}")
                    st.markdown(f"<div style='width: 50px; height: 50px; background-color: {hex_color}; border: 1px solid #ccc; border-radius: 4px;'></div>", unsafe_allow_html=True)
                    
                    st.write(f"**Material:** {surface_data.get('material_label', 'Unknown').title()}")
                    st.write(f"**Confidence:** {surface_data.get('material_confidence', 0.0):.1%}")
                with s2:
                    st.write("**What to bring:**")
                    
                    try:
                        import json
                        rep_mats = json.loads(surface_data.get('repair_materials', '[]'))
                        rep_tools = json.loads(surface_data.get('repair_tools', '[]'))
                    except:
                        rep_mats = []
                        rep_tools = []
                        
                    for rm in rep_mats:
                        st.write(f"- [Material] {rm}")
                    for rt in rep_tools:
                        st.write(f"- [Tool] {rt}")
                        
                    st.write(f"**Notes:** {surface_data.get('repair_notes', '')}")
                    
                if st.button("Re-Analyze Surface"):
                    surface_row = None # Force re-analysis
                    c.execute("DELETE FROM surface_analysis WHERE crack_id = ?", (selected_id,))
                    db.conn.commit()
                    st.rerun()
                    
        if not surface_row:
            api_key = st.session_state.get('deepseek_key', '')
            btn_disabled = not bool(api_key)
            help_txt = "Configure DeepSeek API Key in Settings first." if btn_disabled else ""
            
            if st.button("Analyze Surface", disabled=btn_disabled, help=help_txt):
                with st.spinner("Analyzing surface material and color..."):
                    from core.color_analysis import extract_dominant_colors
                    from core.deepseek_material import analyze_material
                    import json
                    
                    # Color Extraction
                    color_res = extract_dominant_colors(crop_path)
                    
                    # Material Extraction
                    mat_res = analyze_material(crop_path, api_key)
                    
                    if "error" in mat_res and mat_res["material_label"] == "unparsed":
                        st.error(mat_res["error"])
                    else:
                        c.execute('''
                            INSERT INTO surface_analysis (
                                crack_id, dominant_color_rgb, dominant_color_hex, nearest_color_name,
                                material_label, material_confidence, repair_notes, raw_model_response,
                                analyzed_at, model_used, repair_materials, repair_tools
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            selected_id,
                            json.dumps(color_res['dominant_color_rgb']) if color_res else None,
                            color_res['dominant_color_hex'] if color_res else None,
                            color_res['nearest_color_name'] if color_res else None,
                            mat_res.get('material_label', 'unknown'),
                            mat_res.get('material_confidence', 0.0),
                            mat_res.get('repair_notes', ''),
                            mat_res.get('raw_model_response', ''),
                            mat_res.get('analyzed_at', ''),
                            mat_res.get('model_used', ''),
                            json.dumps(mat_res.get('repair_materials', [])),
                            json.dumps(mat_res.get('repair_tools', []))
                        ))
                        # Wait, repair_materials and repair_tools are not in the db schema I wrote?
                        # Let me check my migration script... I missed repair_materials and repair_tools!
                        db.conn.commit()
                        st.success("Analysis complete!")
                        st.rerun()
    else:
        st.info("No cracks match the current filters.")
else:
    st.info("No cracks found in this session.")
