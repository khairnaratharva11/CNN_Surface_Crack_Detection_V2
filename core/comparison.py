import math
from datetime import datetime
from core.config import config

def compute_session_summary(session_id, sessions, cracks):
    """
    Computes summary metrics for a given session.
    sessions: list of session dicts
    cracks: list of crack dicts (for all sessions, or pre-filtered)
    """
    session = next((s for s in sessions if s['id'] == session_id), None)
    if not session:
        return None
        
    session_cracks = [c for c in cracks if c['session_id'] == session_id]
    
    total_cracks = len(session_cracks)
    counts_by_severity = {"Critical": 0, "High": 0, "Moderate": 0, "Low": 0}
    max_score = 0.0
    sum_score = 0.0
    unreviewed_count = 0
    
    for c in session_cracks:
        sev = c.get('severity_level', 'Low')
        if sev in counts_by_severity:
            counts_by_severity[sev] += 1
        
        score = c.get('severity_score', 0.0)
        if score > max_score:
            max_score = score
        sum_score += score
        
        if c.get('status', 'new') == 'new':
            unreviewed_count += 1
            
    mean_score = sum_score / total_cracks if total_cracks > 0 else 0.0
    unreviewed_ratio = unreviewed_count / total_cracks if total_cracks > 0 else 0.0
    
    # Calculate density (cracks per minute, or fallback to something else if duration is 0)
    # The session start_time and end_time are ISO strings
    start_str = session.get('start_time')
    end_str = session.get('end_time')
    
    duration_minutes = 1.0 # default non-zero
    if start_str and end_str:
        try:
            start_dt = datetime.fromisoformat(start_str)
            end_dt = datetime.fromisoformat(end_str)
            duration_minutes = max(0.1, (end_dt - start_dt).total_seconds() / 60.0)
        except ValueError:
            pass
            
    density = total_cracks / duration_minutes
    
    return {
        "session_id": session_id,
        "site_tag": session.get('site_tag', 'Unknown'),
        "total_cracks": total_cracks,
        "counts_by_severity": counts_by_severity,
        "max_severity_score": max_score,
        "mean_severity_score": mean_score,
        "density": density,
        "unreviewed_ratio": unreviewed_ratio,
        "is_calibrated": session.get('mm_per_pixel') is not None
    }

def compute_priority_score(summary):
    if not summary or summary['total_cracks'] == 0:
        return 0.0
        
    cfg = config.get('comparison', {})
    w_max = cfg.get('weight_max_severity', 0.3)
    w_crit = cfg.get('weight_critical_count', 0.4)
    k = cfg.get('critical_k', 2.0)
    w_dens = cfg.get('weight_density', 0.2)
    w_unrev = cfg.get('weight_unreviewed', 0.1)
    
    crit_count = summary['counts_by_severity']['Critical']
    high_count = summary['counts_by_severity']['High']
    
    # Normalize density somewhat to 0-100 range. Say 100 cracks/min is score 100
    density_norm = min(100.0, summary['density'] * 1.0)
    
    # max_severity is 0-100
    # critical component: scale by 10 per weighted count to get a comparable 0-100 score
    crit_score = min(100.0, (crit_count * k + high_count) * 10.0)
    
    unrev_score = summary['unreviewed_ratio'] * 100.0
    
    priority = (w_max * summary['max_severity_score'] + 
                w_crit * crit_score + 
                w_dens * density_norm + 
                w_unrev * unrev_score)
                
    return min(100.0, max(0.0, priority))

def rank_sessions(session_ids, db):
    """
    Reads from db directly to get all sessions and cracks once.
    Returns ranked list of dictionaries.
    """
    sessions = db.get_sessions()
    cracks = db.get_cracks() # get all cracks, filtering done in compute_session_summary
    
    ranked = []
    for sid in session_ids:
        summary = compute_session_summary(sid, sessions, cracks)
        if not summary:
            continue
            
        priority = compute_priority_score(summary)
        
        # Build plain-English justification
        crit = summary['counts_by_severity']['Critical']
        unrev = summary['unreviewed_ratio'] * 100
        justification = (f"{crit} CRITICAL cracks, "
                         f"max severity {summary['max_severity_score']:.1f}, "
                         f"{int(unrev)}% unreviewed.")
                         
        if not summary['is_calibrated']:
            justification += " (Uncalibrated data)"
            
        ranked.append({
            "session_id": sid,
            "site_tag": summary['site_tag'],
            "priority_score": priority,
            "summary": summary,
            "justification": justification
        })
        
    # Sort descending
    ranked.sort(key=lambda x: x['priority_score'], reverse=True)
    return ranked
