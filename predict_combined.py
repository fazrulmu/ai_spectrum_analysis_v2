import os
# Configure for ROCm GPU with fallback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib
import sys
import os
from pathlib import Path
from scipy.interpolate import interp1d

# --- Import Model Architectures ---
# (Ideally these should be in a separate file, but for now we redefine them or import if possible)
# To keep it simple and self-contained, I will redefine the classes here matching the training scripts.

class FullSpectrumCNN(nn.Module):
    def __init__(self, num_classes, input_length=3911):
        super(FullSpectrumCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class VerifierModel(nn.Module):
    def __init__(self, num_groups, embedding_dim=16):
        super(VerifierModel, self).__init__()
        self.spec_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.group_embed = nn.Embedding(num_groups, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(64 + embedding_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x_spec, grp_id):
        feat = self.spec_conv(x_spec).view(x_spec.size(0), -1)
        emb = self.group_embed(grp_id).view(grp_id.size(0), -1)
        return self.classifier(torch.cat([feat, emb], dim=1))

# --- Tiny LLM (Simple Inference) ---
class TinyChemicalLLM:
    def __init__(self, model_path='saved_models/tiny_llm.pth', vocab_path='saved_models/tiny_llm_vocab.json'):
        # Placeholder: In a real scenario, we would load the trained GPT model.
        # Since the user trained it in a notebook, we might not have the .pth handy or the class def here.
        # For this demonstration, I will simulate the LLM response or assume the user exports it.
        # I will implement a simple rule-based fallback if model is missing, 
        # but the structure allows plugging in the real model.
        self.model = None
        print("🤖 Tiny LLM initialized (Simulation Mode for now)")

    def generate(self, prompt):
        # Simulate generation based on prompt
        # Prompt: "User: Group Carbonyl, Alkane?"
        # Output: "Assistant: SMILES ..."
        
        # Simple heuristic for demo
        if "Carbonyl" in prompt and "Alkane" in prompt:
            return "CC(=O)CCC" # Example ketone
        elif "Benzene" in prompt:
            return "c1ccccc1"
        else:
            return "C" # Default

# --- Main Predictor Class ---
class SpectrumPredictor:
    def __init__(self):
        # Enable ROCm GPU if available, otherwise CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✅ PyTorch using ROCm GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("ℹ️ PyTorch using CPU (no GPU detected)")
        self.load_resources()
        
    def load_resources(self):
        print("🔄 Loading models and resources...")
        try:
            # 1. Full Spectrum Model
            self.full_enc = joblib.load('saved_models/full_spectrum_encoder.joblib')
            self.full_model = FullSpectrumCNN(num_classes=len(self.full_enc.classes_)).to(self.device)
            self.full_model.load_state_dict(torch.load('saved_models/full_spectrum_model.pth', map_location=self.device))
            self.full_model.eval()
            
            # 2. Verifier Model
            self.ver_group_map = joblib.load('saved_models/func_group_encoder.joblib')
            self.ver_model = VerifierModel(num_groups=len(self.ver_group_map)).to(self.device)
            self.ver_model.load_state_dict(torch.load('saved_models/func_group_verifier.pth', map_location=self.device))
            self.ver_model.eval()
            
            # 3. Structural Confidence (for ranges)
            # Try multiple locations
            struct_path = 'data/for_train/structural_confidence.json'
            if not os.path.exists(struct_path):
                struct_path = 'saved_models/structural_confidence.json'
                
            with open(struct_path, 'r') as f:
                self.struct_conf = json.load(f)
                
            self.group_ranges = {}
            for cas, info in self.struct_conf.items():
                for grp, det in info.get('detected_functional_groups', {}).items():
                    if grp not in self.group_ranges:
                        self.group_ranges[grp] = {'min': det['range_min'], 'max': det['range_max']}
            
            # 4. LLM
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from src.llm.llm_engine import LLMAnalyzer
                self.llm = LLMAnalyzer()
            except Exception as e:
                print(f"⚠️ LLM Initialization Failed: {e}")
                self.llm = None
            
            print("✅ All models loaded successfully.")
            
        except Exception as e:
            print(f"❌ Error loading resources: {e}")
            print("Did you run both training scripts?")
            sys.exit(1)

    def detect_spectrum_type(self, jdx_path):
        # Simple heuristic based on filename or content
        filename = Path(jdx_path).name.lower()
        if 'uv' in filename: return 'uv'
        if 'ir' in filename: return 'ir'
        
        # Fallback: Read first few lines for data type
        try:
            with open(jdx_path, 'r') as f:
                content = f.read(1000).lower()
                if 'uv-vis' in content or 'absorbance' in content: # Weak check
                    # Check X units
                    if 'nanometers' in content: return 'uv'
                    if 'cm-1' in content: return 'ir'
        except:
            pass
        return 'ir' # Default

    def preprocess(self, raw_data, spec_type='ir'):
        # Reuse the logic from preprocess_universal (simplified)
        # In a real app, import this from src.data.data_processing
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.data.data_processing import preprocess_spectrum
        import yaml
        
        with open('main_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Process
        df = preprocess_spectrum(raw_data, config, spec_type, normalize=True)
        if df is None: return None, None
        
        # Return Absorbance array and Grid
        if spec_type == 'ir':
            return df['absorbance'].values, df['wavenumber'].values
        else:
            return df['log_epsilon'].values, df['wavelength'].values

    def predict(self, jdx_path, spec_type=None, uv_data=None, matrix_type="ATR_NEAT"):
        """Legacy file-based prediction"""
        # 1. Load File
        if spec_type:
             is_ir = (spec_type.lower() == 'ir')
        else:
             is_ir = self.detect_spectrum_type(jdx_path)
        
        # Load config for preprocessing
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.data.data_processing import parse_jdx
        import yaml
        with open('main_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        raw = None
        if jdx_path.lower().endswith('.csv'):
            try:
                df = pd.read_csv(jdx_path, header=None)
                raw = {
                    'x': df.iloc[:, 0].values,
                    'y': df.iloc[:, 1].values,
                    'metadata': {'xunits': 'cm-1', 'yunits': 'absorbance'}
                }
            except: pass
        else:
            raw = parse_jdx(jdx_path)
            
        if not raw:
            return {"error": "Failed to parse file"}

        # 2. Run Core Login
        spec_type_str = 'ir' if is_ir else 'uv'
        return self._core_predict(raw, spec_type_str, matrix_type, uv_data)

    def predict_spectrum(self, x, y, spec_type='ir', matrix_context="ATR_NEAT"):
        """
        GUI-facing method: Accepts arrays, returns formatted dict for GUI Cards.
        """
        # 1. Construct Raw Data
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import yaml
        with open('main_config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        raw = {
            'x': x,
            'y': y,
            'metadata': {
                'xunits': 'cm-1' if spec_type == 'ir' else 'nm',
                'yunits': 'absorbance',
                'matrix_sample': matrix_context
            }
        }
        
        # 2. Run Core Logic
        result = self._core_predict(raw, spec_type, matrix_context)
        
        # 3. Format for GUI (Adapter)
        gui_result = {}
        
        # A. Compound Info
        llm = result.get('llm_analysis', {})
        gui_result['compound_name'] = llm.get('compound_name', 'Unknown Compound')
        gui_result['confidence'] = 0.0
        try:
             conf_str = str(llm.get('confidence', '0.0')).replace('%', '')
             gui_result['confidence'] = float(conf_str) / 100.0 if float(conf_str) > 1.0 else float(conf_str)
        except: pass
        
        gui_result['matrix'] = matrix_context
        
        # B. Highlights (List of Dicts)
        # result['verified_ranges'] = {'Alcohol': [3200, 3600], ...}
        highlights = []
        ranges = result.get('verified_ranges', {})
        
        # Color Palette
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c']
        
        for i, (grp_name, r_vals) in enumerate(ranges.items()):
            color = colors[i % len(colors)]
            highlights.append({
                'label': grp_name,
                'range': list(r_vals),
                'color': color,
                'confidence': 1.0 # Verified
            })
            
        gui_result['highlights'] = highlights
        
        # C. Processed Data (for comparison plot)
        # We need to retrieve the preprocessed x/y used in _core_predict
        # Since _core_predict returns the analysis dict, we might have lost the actual spectrum arrays unless we add them to return.
        # For now, let's assume the GUI plots the Sample Input as Main, and we can plot the CAS match as comparison?
        # Or simpler: The GUI is mostly plotting Sample. The "Predicted/Ref" in canvas usually expects the processed version.
        # Let's re-preprocess just to get the grid if needed, or modify _core_predict to return it.
        # I'll update _core_predict to include 'spectrum_data' in output.
        
        if 'spectrum_data' in result:
             gui_result['processed_y'] = result['spectrum_data']
             # Reconstruct X grid (approx 4000-400)
             gui_result['processed_x'] = np.linspace(4000, 400, len(result['spectrum_data']))
        
        return gui_result

    def _core_predict(self, raw, spec_type, matrix_type, uv_data=None):
        """Internal Pipeline"""
        # Preprocess
        spectrum, grid = self.preprocess(raw, spec_type)
        if spectrum is None: return {"status": "ERROR", "reason": "Preprocessing failed"}
        
        if spec_type == 'uv':
            return self.analyze_uv(spectrum, grid)

        # --- IR LOGIC ---
        # 1. Global Scan
        input_len = 3911
        if len(spectrum) > input_len: x = spectrum[:input_len]
        else: x = np.pad(spectrum, (0, input_len - len(spectrum)), mode='constant')
        x_tensor = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.full_model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        candidates = []
        for idx, prob in enumerate(probs):
            if prob > 0.3:
                grp_name = self.full_enc.classes_[idx]
                candidates.append((grp_name, prob))
                
        # 2. Local Verification
        from scipy.interpolate import interp1d
        verified_groups = []
        for grp, glob_score in candidates:
             if grp not in self.ver_group_map or grp not in self.group_ranges:
                 if glob_score > 0.6: verified_groups.append(grp)
                 continue
                 
             r_min, r_max = self.group_ranges[grp]['min'], self.group_ranges[grp]['max']
             idx_start = max(0, min(int(4000 - r_max), 3910))
             idx_end = max(0, min(int(4000 - r_min), 3910))
             if idx_start >= idx_end: idx_start, idx_end = idx_end, idx_start
             
             y_slice = x[idx_start:idx_end+1]
             if len(y_slice) < 2: y_res = np.zeros(128)
             else:
                 f = interp1d(np.linspace(0, 1, len(y_slice)), y_slice, kind='linear', fill_value="extrapolate")
                 y_res = f(np.linspace(0, 1, 128))
                 
             x_ver = torch.FloatTensor(y_res).unsqueeze(0).unsqueeze(0).to(self.device)
             grp_id = torch.LongTensor([self.ver_group_map[grp]]).to(self.device)
             with torch.no_grad(): ver_score = self.ver_model(x_ver, grp_id).item()
             
             final_score = (0.3 * glob_score) + (0.7 * ver_score)
             if final_score > 0.5: verified_groups.append(grp)

        # 3. Smart Engine
        verified_ranges = {}
        for grp in verified_groups:
             if grp in self.group_ranges: verified_ranges[grp] = [self.group_ranges[grp]['min'], self.group_ranges[grp]['max']]
             
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from src.logic.smart_engine import SmartAnalysisEngine
        smart_engine = SmartAnalysisEngine()
        smart_result = smart_engine.process_sample({
            "id": "Manual_Input",
            "ir_groups": verified_groups,
            "verified_ranges": verified_ranges,
            "uv_data": uv_data,
            "matrix": matrix_type
        })
        filtered_groups = smart_result['filtered_ir_groups']
        # MISSING EXTRACTION LOGIC
        ignored_peaks = smart_result.get('ignored_peaks', [])
        matrix_desc = smart_result.get('active_matrix_desc', '')
        system_note = smart_result['system_note']
        
        # 4. CAS Search
        ir_grid = np.linspace(4000, 90, 3911)
        cas_matches = self.search_cas(x, ir_grid, 'ir', matrix_type=matrix_type)
        
        # 5. LLM
        analysis_result = {}
        if self.llm:
             analysis_result = self.llm.analyze(filtered_groups, "N/A", cas_matches[:3], 'ir')
             
        return {
            "status": "FINALIZED",
            "verified_groups": filtered_groups,
            "verified_ranges": {k:v for k,v in verified_ranges.items() if k in filtered_groups},
            "llm_analysis": analysis_result,
            "spectrum_data": spectrum,
            "cas_matches": cas_matches,
            "ignored_peaks": ignored_peaks,
            "matrix_desc": matrix_desc,
            "system_note": system_note,
            "spectrum_type": "ir"
        }

    def analyze_uv(self, spectrum, grid):
        print("\n🔬 UV-Vis Analysis Mode")
        
        # 1. Find Lambda Max
        max_idx = np.argmax(spectrum)
        lambda_max = grid[max_idx]
        epsilon = spectrum[max_idx] # This is log_epsilon or absorbance depending on preprocess
        
        print(f"📈 Detected Peak: λmax = {lambda_max:.1f} nm | Intensity = {epsilon:.2f}")
        
        # 2. Apply Rules (from todo_uv.md)
        print("\n🧠 Applying Structural Rules:")
        structural_rules = []
        
        if lambda_max < 200:
            msg = "λmax < 200 nm: Likely Isolated C=C or C=O (Non-conjugated)."
            print(f"   - {msg}")
            structural_rules.append(msg)
        elif 200 <= lambda_max <= 250:
            msg = "λmax 200-250 nm: Possible Conjugated System (Diene) or Aromatic (Benzene E-band)."
            print(f"   - {msg}")
            structural_rules.append(msg)
        elif 250 < lambda_max <= 300:
            msg = "λmax 250-300 nm: Likely Aromatic (Benzene B-band) or Carbonyl (n->pi*)."
            print(f"   - {msg}")
            structural_rules.append(msg)
            
            if epsilon < 100: # Weak
                 msg_int = "Low Intensity: Suggests n->pi* transition (Ketone/Aldehyde)."
                 print(f"     -> {msg_int}")
                 structural_rules.append(msg_int)
            else:
                 msg_int = "High Intensity: Suggests pi->pi* transition (Aromatic/Conjugated)."
                 print(f"     -> {msg_int}")
                 structural_rules.append(msg_int)
        else:
            msg = "λmax > 300 nm: Highly Conjugated System or Colored Compound."
            print(f"   - {msg}")
            structural_rules.append(msg)
            
        # 3. Search CAS
        print("\n🔍 Performing UV Spectral Search...")
        # Pass matrix_type='ATR_NEAT' default for UV for now as explicit matrix masking for UV isn't defined yet
        cas_matches = self.search_cas(spectrum, grid, 'uv', matrix_type='ATR_NEAT')
        
        # 4. LLM Analysis
        analysis_result = {}
        if self.llm:
            print("\n🤖 Running LLM Analysis...")
            uv_peaks = f"{lambda_max:.1f} nm (Abs: {epsilon:.2f})"
            top_matches = cas_matches[:3] if cas_matches else []
            
            analysis_result = self.llm.analyze(
                functional_groups=[], # UV doesn't give FG directly usually
                uv_peaks=uv_peaks,
                cas_matches=top_matches,
                spectrum_type='uv'
            )
            
            print(json.dumps(analysis_result, indent=2))
            
        return {
            "verified_groups": [],
            "cas_matches": cas_matches,
            "llm_analysis": analysis_result,
            "spectrum_type": "uv",
            "uv_peak": f"{lambda_max:.1f} nm",
            "lambda_max": float(lambda_max),
            "structural_rules": structural_rules
        }

        if not Path(db_path).exists():
            print(f"⚠️ Training database not found ({db_path}). Skipping CAS search.")
            return [] # Modified to return empty list
            
        candidates = []
        
        # --- SOLVENT MASKING (CRITICAL FIX) ---
        # If matrix_type is provided, zero out invalid regions
        # This prevents solvent peaks (CCl4, CS2) from matching incorrectly.
        masked_spectrum = input_spectrum.copy()
        
        # Use SmartEngine rules if possible, or hardcode here for speed
        # Since we are inside search_cas, let's look at passed kwargs or just implement the rules inline
        # Ideally, we should pass 'matrix_type' to search_cas.
        # Let's assume it was passed in kwargs or added to signature.
        # Current signature: search_cas(self, input_spectrum, input_grid, spec_type='ir', top_k=5, matrix_type='ATR_NEAT')
        
        # To make this robust without changing every call site immediately, we'll check logic below.
        # But wait, search_cas definition above doesn't have matrix_type! 
        # I need to update the signature OR handle it before calling.
        # UPDATING SIGNATURE WOULD BREAK CALLERS IF NOT CAREFUL.
        # Let's inspect where search_cas is called. It is called in predict() lines 365 and 444.
        
        # NOTE: I will update the signature in a separate edit. For now, let's just add the logic 
        # assuming 'matrix_type' is available in a future edit, OR strictly hardcode common solvents if no matrix passed.
        # Actually, I can add matrix_type to the args in this same replacement if I update the def line too.
        
        pass 

    def search_cas(self, input_spectrum, input_grid, spec_type='ir', top_k=5, matrix_type="ATR_NEAT"):
        # Load Database (Lazy Loading)
        if spec_type == 'ir':
            db_path = "data/for_train/universal_training_dataset_IR.jsonl"
        else:
            db_path = "data/for_train/universal_training_dataset_UV.jsonl"
            
        if not Path(db_path).exists():
            print(f"⚠️ Training database not found ({db_path}). Skipping CAS search.")
            return []

        # --- MASKING LOGIC ---
        masked_input = input_spectrum.copy()
        mask_indices = []
        
        # Define ranges (CM-1 for IR)
        avoid_ranges = []
        if spec_type == 'ir':
            if "CCL4" in matrix_type.upper():
                avoid_ranges.append([0, 1350]) # Ignore below 1350
            elif "CS2" in matrix_type.upper():
                avoid_ranges.append([2100, 2200])
                avoid_ranges.append([1400, 1600])
            elif "NUJOL" in matrix_type.upper():
                avoid_ranges.append([2800, 3000])
                avoid_ranges.append([1350, 1480])
        
        # Apply Mask
        if avoid_ranges:
            print(f"   🛡️ Masking Solvent Regions for {matrix_type}: {avoid_ranges}")
            for start, end in avoid_ranges:
                # Find indices in input_grid
                # Grid is likely descending 4000->400 for IR
                # Condition: start <= grid <= end
                # Use numpy for speed
                mask = (input_grid >= start) & (input_grid <= end)
                masked_input[mask] = 0.0

        candidates = []
        
        # Flatten input
        input_vec = masked_input.flatten() # Use MASKED input
        norm_input = np.linalg.norm(input_vec)
        if norm_input == 0: norm_input = 1e-10
        
        try:
            with open(db_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        db_y = np.array(record['y'])
                        
                        # ALIGNMENT FIX: Use 'x' from record to interpolate onto input_grid
                        if 'x' in record:
                            db_x = np.array(record['x'])
                            
                            # Check if interpolation is needed
                            # If lengths differ OR if we suspect grid mismatch
                            # It's safest to ALWAYS interpolate to be sure we compare same wavelengths
                            
                            # Handle direction (interp1d expects sorted x)
                            if db_x[0] > db_x[-1]:
                                db_x = db_x[::-1]
                                db_y = db_y[::-1]
                                
                            f_interp = interp1d(db_x, db_y, kind='linear', bounds_error=False, fill_value=0.0)
                            db_y_aligned = f_interp(input_grid)
                            
                        else:
                            # Fallback to index-based if 'x' missing (should not happen for our data)
                            if len(db_y) != len(input_vec):
                                f_interp = interp1d(np.linspace(0, 1, len(db_y)), db_y, kind='linear')
                                db_y_aligned = f_interp(np.linspace(0, 1, len(input_vec)))
                            else:
                                db_y_aligned = db_y

                        # Cosine Similarity
                        norm_db = np.linalg.norm(db_y_aligned)
                        if norm_db == 0: norm_db = 1e-10
                        
                        score = np.dot(input_vec, db_y_aligned) / (norm_input * norm_db)
                        
                        # Get Metadata if available
                        meta = record.get('metadata', {})
                        l_max = meta.get('lambda_max', 'N/A')
                        
                        candidates.append({
                            'cas': record['cas_number'],
                            'smiles': record.get('smiles', 'N/A'),
                            'score': score,
                            'l_max': l_max
                        })
                    except Exception as e:
                        # print(f"Debug error: {e}")
                        continue
                        
            # Sort and Print
            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"🏆 Top {top_k} CAS Matches ({spec_type.upper()}):")
            for i, match in enumerate(candidates[:top_k]):
                l_max_str = f" | λmax: {match['l_max']}" if spec_type == 'uv' else ""
                print(f"   {i+1}. CAS: {match['cas']} | Score: {match['score']:.4f} | SMILES: {match['smiles']}{l_max_str}")
            
            return candidates # Modified to return candidates
                
        except Exception as e:
            print(f"❌ Search Error: {e}")
            return [] # Modified to return empty list

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_combined.py <path_to_jdx>")
    else:
        predictor = SpectrumPredictor()
        predictor.predict(sys.argv[1])
