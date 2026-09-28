import streamlit as st
import yaml
import os
from core.config import config

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("System Settings")

st.markdown("Update thresholds, severity bands, and camera calibration. Click Save to apply (this will restart the app session).")

calib = st.number_input("Calibration (pixels per mm)", value=config['camera'].get('calibration_pixels_per_mm') or 0.0, format="%.2f")

st.subheader("Severity Thresholds (Length)")
l_low = st.number_input("Low Threshold", value=config['severity']['length_low'])
l_mod = st.number_input("Moderate Threshold", value=config['severity']['length_moderate'])
l_high = st.number_input("High Threshold", value=config['severity']['length_high'])

if st.button("Save Settings"):
    # Read yaml, update, write
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(root_dir, 'config.yaml')
    
    with open(config_path, 'r') as f:
        full_cfg = yaml.safe_load(f)
        
    full_cfg['camera']['calibration_pixels_per_mm'] = calib if calib > 0 else None
    full_cfg['severity']['length_low'] = l_low
    full_cfg['severity']['length_moderate'] = l_mod
    full_cfg['severity']['length_high'] = l_high
    
    with open(config_path, 'w') as f:
        yaml.dump(full_cfg, f, default_flow_style=False)
        
    st.success("Settings saved! Please restart or refresh the application.")

st.markdown("---")
st.subheader("DeepSeek Vision Settings")
st.markdown("Configure the AI material and color analysis module. *Key is used only for this session and is never saved.*")

ds_key = st.text_input("DeepSeek API Key", value=st.session_state.get('deepseek_key', ''), type="password")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Set Key for Session"):
        st.session_state['deepseek_key'] = ds_key
        st.success("Key set in session memory.")
with col_b:
    if st.button("Test Connection"):
        if 'deepseek_key' not in st.session_state or not st.session_state['deepseek_key']:
            st.warning("Please set the API key first.")
        else:
            import requests
            try:
                # Make a minimal test call (not image, just text to check auth)
                resp = requests.post("https://api.deepseek.com/chat/completions",
                                    headers={"Authorization": f"Bearer {st.session_state['deepseek_key']}", "Content-Type": "application/json"},
                                    json={"model": "deepseek-flash", "messages": [{"role": "user", "content": "ping"}]},
                                    timeout=10)
                if resp.status_code == 200:
                    st.success("Connection successful!")
                else:
                    st.error(f"Failed: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Error connecting: {str(e)}")

auto_analyze = st.toggle("Auto-analyze new cracks", value=config.get('deepseek', {}).get('auto_analyze', False))

if st.button("Save DeepSeek Preferences"):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(root_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        full_cfg = yaml.safe_load(f)
    
    if 'deepseek' not in full_cfg:
        full_cfg['deepseek'] = {}
    full_cfg['deepseek']['auto_analyze'] = auto_analyze
    
    with open(config_path, 'w') as f:
        yaml.dump(full_cfg, f, default_flow_style=False)
        
    st.success("Preferences saved!")
