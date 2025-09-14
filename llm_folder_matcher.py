import os
import json
from rapidfuzz import fuzz
import torch
from sentence_transformers import SentenceTransformer, util

class LLMFolderClassifier:
    def __init__(self):
        self.reference_map = self._load_reference_data()

    def _load_reference_data(self):
        ref_file = os.path.join(os.path.dirname(__file__), "reference_descriptions.json")
        with open(ref_file, "r") as f:
            return json.load(f)

    def classify(self, prompt):
        if not self.reference_map:
            print(f"[Prompt Lora Matcher] No reference descriptions found in reference_descriptions.json")
            return None

        print(f"[Prompt Lora Matcher] Searching for LoRA match among {len(self.reference_map)} available LoRAs...")
        best_score = -1
        best_path = None

        for path, desc in self.reference_map.items():
            # Use fuzzy string matching to compare prompt with description
            fuzzy_score = fuzz.token_sort_ratio(prompt.lower(), desc.lower()) / 100

            if fuzzy_score > best_score:
                best_score = fuzzy_score
                best_path = path

        print(f"[Prompt Lora Matcher] Best match score: {best_score:.3f} for '{best_path}'")

        # Only return a match if the score is above a reasonable threshold (0.3 = 30% similarity)
        if best_score > 0.3:
            return best_path
        else:
            print(f"[Prompt Lora Matcher] Best match score {best_score:.3f} below threshold (0.3). No LoRA will be loaded.")
            return None


class LLMFolderClassifierAI:
    def __init__(self):
        self.embedder = None
        self.reference_map = self._load_reference_data()
        self.reference_embeddings = {}
        self._check_model_availability()

    def _load_reference_data(self):
        ref_file = os.path.join(os.path.dirname(__file__), "reference_descriptions.json")
        with open(ref_file, "r") as f:
            return json.load(f)

    def _check_model_availability(self):
        """Check if AI models are available and provide warnings/instructions if not."""
        model_dir = os.path.join(os.path.dirname(__file__), "model")

        if not os.path.exists(model_dir):
            self._show_model_warning("create_model_dir")
            return

        # Check for sentence transformer models (local files)
        st_models = []
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith(('.bin', '.safetensors', '.pt', '.pth')) or 'sentence-transformer' in file.lower():
                    st_models.append(os.path.join(root, file))

        # Check for GGUF models (for potential future LLM integration)
        gguf_models = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]

        if not st_models and not gguf_models:
            self._show_model_warning("download_models")
        elif not st_models and gguf_models:
            self._show_model_warning("sentence_transformer_needed")
        else:
            print(f"[AI LoRA Matcher] Found {len(st_models)} AI model(s) in model directory")
            if gguf_models:
                print(f"[AI LoRA Matcher] Also found {len(gguf_models)} GGUF model(s)")

    def _show_model_warning(self, instruction_key):
        """Display warning messages and instructions for missing models."""
        warnings = {
            "create_model_dir": """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ⚠️  MODEL DIRECTORY MISSING ⚠️                      ║
║                                                                            ║
║  The AI LoRA Matcher requires a 'model' directory but it's missing.      ║
║                                                                            ║
║  SOLUTION:                                                                ║
║  1. Create the directory: custom_nodes/ComfyUI-LoraPromptMatcher/model/  ║
║  2. Download sentence-transformer models (see instructions below)        ║
║                                                                            ║
║  The node will still work but will download models from HuggingFace      ║
║  automatically (slower first run).                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """,

            "download_models": """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🤖 AI MODELS NOT FOUND 🤖                          ║
║                                                                            ║
║  The AI LoRA Matcher requires sentence-transformer models for semantic    ║
║  matching, but none were found in the model directory.                    ║
║                                                                            ║
║  RECOMMENDED MODELS:                                                      ║
║  • all-MiniLM-L12-v2 (DEFAULT - higher accuracy, medium speed)           ║
║  • all-MiniLM-L6-v2 (fast, good balance)                                   ║
║  • paraphrase-MiniLM-L6-v2 (good for paraphrases)                         ║
║                                                                            ║
║  DOWNLOAD FROM: huggingface.co/sentence-transformers/                     ║
║  Place model files in: custom_nodes/ComfyUI-LoraPromptMatcher/model/      ║
║                                                                            ║
║  ALTERNATIVE: The node will automatically download models from           ║
║  HuggingFace on first use (may be slower).                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """,

            "sentence_transformer_needed": """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🔄 GGUF MODELS FOUND - ST MODELS NEEDED 🔄                ║
║                                                                            ║
║  Found GGUF models but the AI LoRA Matcher needs sentence-transformer     ║
║  models for semantic similarity matching.                                  ║
║                                                                            ║
║  The GGUF models you have are for local LLM inference, but this node      ║
║  uses sentence-transformers for text similarity.                          ║
║                                                                            ║
║  DOWNLOAD sentence-transformer models from:                               ║
║  huggingface.co/sentence-transformers/                                     ║
║                                                                            ║
║  Recommended: all-MiniLM-L12-v2 (higher accuracy)                          ║
║  Place in: custom_nodes/ComfyUI-LoraPromptMatcher/model/                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
            """
        }

        warning_msg = warnings.get(instruction_key, "Unknown warning type")
        print(warning_msg)

        # Also show a shorter console message
        if instruction_key == "create_model_dir":
            print("[AI LoRA Matcher] ⚠️  WARNING: Model directory missing!")
            print("[AI LoRA Matcher] 📁 Create: custom_nodes/ComfyUI-LoraPromptMatcher/model/")
        elif instruction_key == "download_models":
            print("[AI LoRA Matcher] ⚠️  WARNING: No AI models found in model directory!")
            print("[AI LoRA Matcher] 📥 Download sentence-transformer models to enable faster local inference")
        elif instruction_key == "sentence_transformer_needed":
            print("[AI LoRA Matcher] ℹ️  INFO: GGUF models found but sentence-transformer models needed")
            print("[AI LoRA Matcher] 📥 Download ST models for optimal performance")

    def _get_embedder(self):
        if self.embedder is None:
            print("[AI LoRA Matcher] Loading sentence transformer model...")
            print("[AI LoRA Matcher] Using all-MiniLM-L12-v2 (higher accuracy model)")
            self.embedder = SentenceTransformer("all-MiniLM-L12-v2")
            print("[AI LoRA Matcher] Model loaded successfully")
        return self.embedder

    def _get_embedding(self, text):
        embedder = self._get_embedder()
        return embedder.encode(text, convert_to_tensor=True)

    def classify(self, prompt):
        if not self.reference_map:
            print("[AI LoRA Matcher] No reference descriptions found in reference_descriptions.json")
            return None

        print(f"[AI LoRA Matcher] Searching for LoRA match among {len(self.reference_map)} available LoRAs...")

        # Get embedding for the input prompt
        prompt_embedding = self._get_embedding(prompt)

        best_score = -1
        best_path = None

        for path, desc in self.reference_map.items():
            # Get or compute embedding for the description
            if path not in self.reference_embeddings:
                self.reference_embeddings[path] = self._get_embedding(desc)

            ref_embedding = self.reference_embeddings[path]

            # Calculate semantic similarity
            similarity = util.pytorch_cos_sim(prompt_embedding, ref_embedding).item()

            # Also calculate fuzzy score for backup
            fuzzy_score = fuzz.token_sort_ratio(prompt.lower(), desc.lower()) / 100

            # Combine semantic and fuzzy scores (weighted average)
            combined_score = (similarity * 0.8) + (fuzzy_score * 0.2)

            if combined_score > best_score:
                best_score = combined_score
                best_path = path

        print(f"[AI LoRA Matcher] Best match score: {best_score:.3f} for '{best_path}'")

        # Lower threshold for AI matching since semantic similarity can be more accurate
        if best_score > 0.2:
            return best_path
        else:
            print(f"[AI LoRA Matcher] Best match score {best_score:.3f} below threshold (0.2). No LoRA will be loaded.")
            return None
