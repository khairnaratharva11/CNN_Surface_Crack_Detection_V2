import sqlite3
import os
import json
from datetime import datetime

class Database:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()
        
    def create_tables(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                source_type TEXT,
                source_path TEXT,
                site_tag TEXT,
                operator TEXT,
                mm_per_pixel REAL,
                model_version TEXT,
                settings_snapshot TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS cracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                crack_id INTEGER,
                first_seen_frame INTEGER,
                first_seen_timestamp REAL,
                last_seen_timestamp REAL,
                bbox TEXT,
                position_in_frame TEXT,
                confidence REAL,
                length REAL,
                mean_width REAL,
                max_width REAL,
                area_ratio REAL,
                unit TEXT,
                severity_score REAL,
                severity_level TEXT,
                recommended_action TEXT,
                snapshot_path TEXT,
                cropped_path TEXT,
                gps_lat REAL,
                gps_lon REAL,
                notes TEXT,
                status TEXT DEFAULT 'new',
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        ''')
        self.conn.commit()

    def start_session(self, session_id, source_type, source_path, site_tag, operator, mm_per_pixel, model_version, settings):
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO sessions 
            (id, start_time, source_type, source_path, site_tag, operator, mm_per_pixel, model_version, settings_snapshot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, datetime.now().isoformat(), source_type, source_path, site_tag, operator, mm_per_pixel, model_version, json.dumps(settings)))
        self.conn.commit()
        
    def end_session(self, session_id):
        c = self.conn.cursor()
        c.execute('UPDATE sessions SET end_time = ? WHERE id = ?', (datetime.now().isoformat(), session_id))
        self.conn.commit()
        
    def upsert_crack(self, session_id, crack_data):
        c = self.conn.cursor()
        
        # Check if crack_id exists for this session
        c.execute('SELECT id, max_width, severity_score FROM cracks WHERE session_id = ? AND crack_id = ?', 
                 (session_id, crack_data['crack_id']))
        row = c.fetchone()
        
        bbox_str = json.dumps(crack_data['bbox'])
        
        if row:
            # Update if the new observation is worse (higher severity score)
            db_id, current_max_width, current_score = row
            if crack_data['severity_stats']['score'] > current_score:
                c.execute('''
                    UPDATE cracks 
                    SET last_seen_timestamp = ?, bbox = ?, position_in_frame = ?, confidence = ?, 
                        length = ?, mean_width = ?, max_width = ?, area_ratio = ?, 
                        severity_score = ?, severity_level = ?, recommended_action = ?, snapshot_path = ?
                    WHERE id = ?
                ''', (
                    crack_data['timestamp'], bbox_str, crack_data['position_in_frame'], crack_data['confidence'],
                    crack_data['severity_stats']['length'], crack_data['severity_stats']['mean_width'],
                    crack_data['severity_stats']['max_width'], crack_data['severity_stats']['area_ratio'],
                    crack_data['severity_stats']['score'], crack_data['severity_stats']['severity_class'],
                    crack_data['severity_stats']['action'], crack_data.get('snapshot_path', ''), db_id
                ))
            else:
                # Just update last_seen
                c.execute('UPDATE cracks SET last_seen_timestamp = ? WHERE id = ?', (crack_data['timestamp'], db_id))
        else:
            # Insert new
            c.execute('''
                INSERT INTO cracks 
                (session_id, crack_id, first_seen_frame, first_seen_timestamp, last_seen_timestamp,
                 bbox, position_in_frame, confidence, length, mean_width, max_width, area_ratio, unit,
                 severity_score, severity_level, recommended_action, snapshot_path, cropped_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, crack_data['crack_id'], crack_data['frame_idx'], crack_data['timestamp'], crack_data['timestamp'],
                bbox_str, crack_data['position_in_frame'], crack_data['confidence'],
                crack_data['severity_stats']['length'], crack_data['severity_stats']['mean_width'],
                crack_data['severity_stats']['max_width'], crack_data['severity_stats']['area_ratio'],
                crack_data['severity_stats']['unit'], crack_data['severity_stats']['score'],
                crack_data['severity_stats']['severity_class'], crack_data['severity_stats']['action'],
                crack_data.get('snapshot_path', ''), crack_data.get('cropped_path', '')
            ))
            
        self.conn.commit()

    def get_cracks(self, filters=None):
        query = "SELECT * FROM cracks"
        params = []
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(f"{k} = ?")
                params.append(v)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        c = self.conn.cursor()
        c.execute(query, params)
        # return as dicts
        cols = [description[0] for description in c.description]
        return [dict(zip(cols, row)) for row in c.fetchall()]
        
    def get_sessions(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM sessions ORDER BY start_time DESC")
        cols = [description[0] for description in c.description]
        return [dict(zip(cols, row)) for row in c.fetchall()]
        
    def update_crack_status(self, crack_db_id, status, notes):
        c = self.conn.cursor()
        c.execute('UPDATE cracks SET status = ?, notes = ? WHERE id = ?', (status, notes, crack_db_id))
        self.conn.commit()
