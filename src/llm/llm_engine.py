import json
import os
from llama_cpp import Llama

class MockLLM:
    def create_chat_completion(self, messages, response_format=None, temperature=0.2, max_tokens=512):
        # Extract user content to generate relevant mock response
        user_content = messages[1]['content']
        
        # Simple heuristic generation
        compound_name = "Unknown Compound"
        confidence = "Low"
        reasoning = "Insufficient data for definitive identification."
        
        if "Benzene" in user_content or "Aromatic" in user_content:
            compound_name = "Benzene Derivative"
            confidence = "Medium"
            reasoning = "Presence of aromatic peaks suggests a benzene ring structure."
            
        if "Carbonyl" in user_content:
            reasoning += " Carbonyl group detected, possibly a ketone or aldehyde."
            
        return {
            'choices': [{
                'message': {
                    'content': json.dumps({
                        "compound_name": compound_name,
                        "confidence": confidence,
                        "reasoning": f"[SIMULATED LLM] {reasoning}",
                        "functional_groups_analysis": "Detected groups are consistent with the proposed structure.",
                        "uv_analysis": "UV peaks indicate conjugation."
                    })
                }
            }]
        }

class LLMAnalyzer:
    def __init__(self, model_path="saved_models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf", n_ctx=2048):
        # Load Knowledge Base
        try:
            kb_path = os.path.join("data", "knowledge_base.json")
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    self.knowledge_base = json.load(f)
                print(f"📚 Knowledge Base Loaded: {len(self.knowledge_base)} entries.")
            else:
                self.knowledge_base = []
                print("⚠️ Knowledge Base file not found.")
        except Exception as e:
            print(f"⚠️ Error loading Knowledge Base: {e}")
            self.knowledge_base = []

        self.model_path = model_path
        if not os.path.exists(model_path):
            print(f"⚠️ LLM Model not found at {model_path}.")
            print("⚠️ Switching to SIMULATION MODE (Mock LLM).")
            self.llm = MockLLM()
        else:
            print(f"🤖 Loading LLM from {model_path}...")
            try:
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=0, 
                    verbose=False
                )
                print("✅ LLM Loaded.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"❌ Failed to load Llama: {e}")
                print("⚠️ Switching to SIMULATION MODE (Mock LLM).")
                self.llm = MockLLM()

    def analyze(self, functional_groups, uv_peaks, cas_matches, spectrum_type='ir'):
        if not self.llm:
            return {"error": "Model not loaded"}

        # Construct Prompt
        system_prompt = """You are an expert spectroscopist. Analyze the provided spectral data (Spectrum type, Detected Groups, UV Peaks, CAS Matches) to identify the compound.
        
        Output must be a single valid JSON object. Do not include any other text.
        
        Example Response Format (Do NOT copy this content, use the structure only):
        {
            "compound_name": "Identified Compound Name",
            "confidence": "High/Medium/Low",
            "reasoning": "Reasoning based on the input data and [KNOWLEDGE BASE FACT CHECK].",
            "functional_groups_analysis": "Relevant groups from input.",
            "uv_analysis": "UV observation.",
            "final_conclusion": "Brief summary (1-2 sentences) of the findings."
        }

        Rules:
        1. Analyze the 'CAS Matches' and 'Detected Groups' provided in the User Input.
        2. Pay extremely close attention to the [KNOWLEDGE BASE FACT CHECK] section if present.
        3. If the [KNOWLEDGE BASE FACT CHECK] says validation FAILED or is Doubted, lower your confidence and explain why in 'reasoning'.
        4. "final_conclusion" must be a concise executive summary.
        5. Return JSON only.
        """
        
        # Limit functional groups to top 5 and format clearly
        top_groups = functional_groups[:5] if functional_groups else []
        groups_str = ", ".join(top_groups)
        
        # --- KNOWLEDGE BASE VALIDATION LOGIC ---
        validation_context = ""
        if cas_matches and len(cas_matches) > 0:
            top_candidate = cas_matches[0]
            top_cas_id = top_candidate.get("cas", "")
            
            # Find in KB
            target_kb = next((item for item in self.knowledge_base if item["cas"] == top_cas_id), None)
            
            if target_kb:
                validation_context = f"""
        [KNOWLEDGE BASE FACT CHECK]
        Target Compound: {target_kb['name']} (CAS {target_kb['cas']})
        Benchmarks:
        - Must have IR peaks: {', '.join(target_kb['ir_markers'])}
        - UV Expectation: {target_kb['uv_expected']}
        - Insight: {target_kb['ai_insight']}
        
        INSTRUCTION: Compare 'Detected Groups' ({groups_str}) against 'Benchmarks'.
        - If essential markers are missing, report 'Reasoning' as DOUBTFUL.
        - If markers match, CONFIRM identification.
        """
        
        user_content = f"""
        Spectrum: {spectrum_type.upper()}
        
        Detected Groups: {groups_str}
        UV Peaks: {uv_peaks if uv_peaks else 'None'}
        CAS Matches: {json.dumps(cas_matches[:2])}
        {validation_context}
        
        Identify the compound based on the above data. Ensure to provide a 'final_conclusion'. Return JSON only.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=256,
                frequency_penalty=0.5,
                presence_penalty=0.0
            )
            
            content = response['choices'][0]['message']['content']
            print(f"🔍 Raw LLM Output: {content}") # Debug
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Fallback: Try to clean markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                    result = json.loads(content)
                else:
                    raise

            # Validation: Ensure 'reasoning' exists
            if 'reasoning' not in result or not result['reasoning']:
                 result['reasoning'] = "Reasoning not provided by AI."
            
            return result
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            if 'content' in locals():
                print(f"Raw Content: {content}")
            return {"error": str(e)}
