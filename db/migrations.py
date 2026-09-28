import sqlite3
import os

def migrate_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS surface_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crack_id INTEGER,
            dominant_color_rgb TEXT,
            dominant_color_hex TEXT,
            nearest_color_name TEXT,
            material_label TEXT,
            material_confidence REAL,
            repair_notes TEXT,
            raw_model_response TEXT,
            analyzed_at TEXT,
            model_used TEXT,
            FOREIGN KEY(crack_id) REFERENCES cracks(id)
        )
    ''')
    
    # Check if columns exist, if not add them
    c.execute("PRAGMA table_info(surface_analysis)")
    columns = [col[1] for col in c.fetchall()]
    
    if "repair_materials" not in columns:
        c.execute("ALTER TABLE surface_analysis ADD COLUMN repair_materials TEXT")
    if "repair_tools" not in columns:
        c.execute("ALTER TABLE surface_analysis ADD COLUMN repair_tools TEXT")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.config import config
    db_path = config['database']['path']
    migrate_db(db_path)
    print("Migration successful.")
