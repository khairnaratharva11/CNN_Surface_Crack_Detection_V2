import os
import pandas as pd
from fpdf import FPDF
from db.database import Database

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'CNN Surface Crack Detection - Inspection Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        # Disclaimer
        self.set_y(-10)
        self.cell(0, 10, 'DISCLAIMER: Model assistance tool; not a substitute for certified structural inspection.', 0, 0, 'C')

def generate_pdf_report(session_id, db_path, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    db = Database(db_path)
    
    sessions = db.get_sessions()
    session = next((s for s in sessions if s['id'] == session_id), None)
    if not session:
        return None
        
    cracks = db.get_cracks({'session_id': session_id})
    df = pd.DataFrame(cracks)
    
    pdf = PDFReport()
    pdf.add_page()
    
    # Session Info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Session Information', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f"Session ID: {session_id}", 0, 1)
    pdf.cell(0, 8, f"Date: {session['start_time']}", 0, 1)
    pdf.cell(0, 8, f"Site Tag: {session['site_tag']}", 0, 1)
    pdf.cell(0, 8, f"Operator: {session['operator']}", 0, 1)
    
    calib = "Yes" if session['mm_per_pixel'] else "No (Relative Pixels)"
    pdf.cell(0, 8, f"Calibrated: {calib}", 0, 1)
    pdf.ln(5)
    
    # Summary
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Summary Counts', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    if not df.empty:
        counts = df['severity_level'].value_counts()
        for level, count in counts.items():
            pdf.cell(0, 8, f"{level}: {count}", 0, 1)
    else:
        pdf.cell(0, 8, "No cracks detected.", 0, 1)
        
    pdf.ln(5)
    
    # Top Critical Cracks
    if not df.empty:
        critical = df[df['severity_level'] == 'Critical']
        if not critical.empty:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Critical Cracks Detail', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for _, row in critical.iterrows():
                pdf.cell(0, 8, f"Crack ID {row['crack_id']} | Score: {row['severity_score']:.1f} | Width: {row['max_width']:.1f}{row['unit']}", 0, 1)
                pdf.cell(0, 8, f"Position: {row['position_in_frame']} | Action: {row['recommended_action']}", 0, 1)
                
                crop_path = row['cropped_path']
                if crop_path and os.path.exists(crop_path):
                    # add image
                    try:
                        pdf.image(crop_path, w=50)
                    except:
                        pass
                pdf.ln(5)

    out_path = os.path.join(output_dir, f"report_{session_id}.pdf")
    pdf.output(out_path)
    return out_path

def export_csv(session_id, db_path, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    db = Database(db_path)
    cracks = db.get_cracks({'session_id': session_id})
    df = pd.DataFrame(cracks)
    out_path = os.path.join(output_dir, f"export_{session_id}.csv")
    df.to_csv(out_path, index=False)
    return out_path
