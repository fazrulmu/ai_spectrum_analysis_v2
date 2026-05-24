
import json

class SmartAnalysisEngine:
    def __init__(self):
        # RULE BASE: Triggers for subsequent analysis
        self.rules = {
            "Ar (C-H Bending)": "CHECK_UV_CONJUGATION",
            "Ar (C-H Stretch)": "CHECK_UV_CONJUGATION",
            "Vinyl (C=C)": "CHECK_UV_CONJUGATION",
            "C=O (Carbonyl)": "CHECK_UV_ABSORPTION" # Optional
        }
        
        # MATRIX BLIND SPOTS / RULES
        self.matrix_rules = {
            "ATR_NEAT": {
                "avoid_ranges": [],
                "forbidden_groups": [],
                "description": "Metode murni. Semua area valid."
            },
            "CCL4": {
                "avoid_ranges": [[0, 900]], # Broad mask for Halogens < 900
                "forbidden_groups": ["chloro", "bromo", "iodo", "fluoro", "halogen", "alkyl halide"],
                "critical_peaks": [[700, 850]], 
                "description": "Abaikan area fingerprint bawah < 900. Hapus Halogen."
            },
            "COMPOSITE": {
                 # For Composite (CCl4 + CS2), we assume it's valid spectrum BUT we must be wary of artifacts.
                 # Given the user's data shows artifacts, we aggressively filter them.
                 "avoid_ranges": [[2100, 2300], [0, 850]], 
                 "forbidden_groups": ["nitril", "alkyne", "cyano", "triple bond", "chloro", "bromo", "iodo", "halo"],
                 "description": "Composite CCl4/CS2. Memblokir Nitril (artifact CS2) dan Halogen (artifact CCl4)."
            },
            "CS2": {
                "avoid_ranges": [[2100, 2300], [1400, 1600]],
                "forbidden_groups": ["nitril", "alkyne", "cyano", "triple bond"],
                "description": "Abaikan area 2100-2300 (Nitril/Alkyne) dan 1400-1600."
            },
            "NUJOL": {
                "avoid_ranges": [[2800, 3000], [1450, 1470], [1370, 1380]],
                "forbidden_groups": ["alkane", "methyl", "methylene"],
                "description": "Abaikan area C-H Alkana (Nujol)."
            },
            "KBR": {
                "avoid_ranges": [[3300, 3500], [1630, 1640]],
                "forbidden_groups": ["alcohol", "hydroxyl", "water"],
                "description": "Abaikan puncak air/OH hygroscopic."
            }
        }

    def process_sample(self, sample_data):
        """
        sample_data: Dict containing 'ir_groups', 'uv_data', 'id', 'matrix', 'verified_ranges'
        """
        detected_ir_groups = sample_data.get('ir_groups', [])
        verified_ranges = sample_data.get('verified_ranges', {}) # Dict: {Group: [Start, End]}
        uv_data = sample_data.get('uv_data', None)
        matrix = sample_data.get('matrix', 'ATR_NEAT').upper()
        
        # --- 1. FILTERING BASED ON MATRIX (BLIND SPOTS) ---
        filtered_groups = []
        ignored_peaks = []
        
        # Normalize matrix key (handle 'SOLUTION (CCL4)' etc if needed, but for now exact match or simplified)
        # Simple heuristic mapping
        active_matrix_rule = None
        for key in self.matrix_rules:
            if key in matrix:
                active_matrix_rule = self.matrix_rules[key]
                break
        
        if not active_matrix_rule:
             active_matrix_rule = self.matrix_rules["ATR_NEAT"] # Default

        avoid_ranges = active_matrix_rule.get("avoid_ranges", [])
        forbidden_keywords = active_matrix_rule.get("forbidden_groups", [])
        
        for group in detected_ir_groups:
            is_ignored = False
            
            # 1. Check forbidden keywords (Kill Switch)
            for keyword in forbidden_keywords:
                if keyword.lower() in group.lower():
                    is_ignored = True
                    ignored_peaks.append(f"{group} (Forbidden '{keyword}' in {matrix})")
                    break
            
            if is_ignored:
                continue

            # 2. Check if this group's range falls into an avoided range
            # Check if this group's range falls into an avoided range
            # We need the range of the detected group.
            # If 'verified_ranges' is provided, use it. Otherwise, rely on name implication (risky) or accept all.
            
            group_range = verified_ranges.get(group)
            is_ignored = False
            
            if group_range:
                g_start, g_end = group_range
                g_center = (g_start + g_end) / 2
                
                for avoid_start, avoid_end in avoid_ranges:
                    # Logic: If the center of the peak is inside the avoid range, ignore it.
                    # Or if there is significant overlap.
                     if avoid_start <= g_center <= avoid_end:
                         is_ignored = True
                         ignored_peaks.append(f"{group} ({g_center:.0f} cm⁻¹) due to {matrix}")
                         break
            
            if not is_ignored:
                filtered_groups.append(group)

        # --- 2. LOGIC TRIGGERS (Smart Decision) ---
        triggers_activated = []
        for group in filtered_groups: # Use filtered groups to trigger!
            for key in self.rules:
                if key in group:
                    triggers_activated.append(self.rules[key])
        
        triggers_activated = list(set(triggers_activated))

        # --- 3. EXECUTION STATE ---
        status = "UNKNOWN"
        message = ""
        action_required = None
        
        if not triggers_activated:
            status = "FINALIZED"
            message = "Analisis IR selesai. Tidak ditemukan gugus aromatik/konjugasi yang memerlukan konfirmasi UV."
        
        elif "CHECK_UV_CONJUGATION" in triggers_activated:
            if uv_data is None:
                status = "HALTED"
                action_required = "EXECUTE_UV_TEST"
                message = "Terdeteksi Cincin Aromatik/Vinil. Analisis DITAHAN. Harap lakukan uji UV untuk konfirmasi konjugasi."
            else:
                status = "FINALIZED"
                uv_peak = uv_data.get('lambda_max', 0)
                if uv_peak > 230: # General rule of thumb
                     message = f"KONFIRMASI POSITIF: IR mendeteksi Aromatik dan UV ({uv_peak:.1f} nm) mengonfirmasi konjugasi."
                else:
                     message = f"KONFLIK DATA: IR mendeteksi Aromatik tetapi UV lemah ({uv_peak:.1f} nm). Mungkin cincin terisolasi."

        return {
            "sample_id": sample_data.get('id'),
            "status": status,
            "action": action_required,
            "system_note": message,
            "filtered_ir_groups": filtered_groups,
            "ignored_peaks": ignored_peaks,
            "active_matrix_desc": active_matrix_rule.get("description", "")
        }
