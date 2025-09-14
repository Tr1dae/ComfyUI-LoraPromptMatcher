from .llm_folder_matcher import LLMFolderClassifier, LLMFolderClassifierAI
import os
import folder_paths
import comfy.sd
import comfy.utils


# Node class definition for ComfyUI
class PromptLoraMatcher:
    NAME = "PromptLoraMatcher"

    def __init__(self):
        self.loaded_lora = None
        self.classifier = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description to match against LoRA descriptions"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_matched_lora"
    CATEGORY = "loaders"

    def load_matched_lora(self, model, prompt, strength_model):
        print(f"[Prompt Lora Matcher] Input prompt: '{prompt}'")

        # Initialize classifier if not already done
        if self.classifier is None:
            self.classifier = LLMFolderClassifier()

        # Find the best matching lora
        matched_lora_name = self.classifier.classify(prompt)

        if matched_lora_name is None:
            print(f"[Prompt Lora Matcher] No matching LoRA found for prompt. Returning original model.")
            return (model,)

        print(f"[Prompt Lora Matcher] Matched LoRA: {matched_lora_name}")

        # Load and apply the matched lora
        if strength_model == 0:
            print(f"[Prompt Lora Matcher] Strength is 0, returning original model.")
            return (model,)

        print(f"[Prompt Lora Matcher] Applying with strength: {strength_model}")

        lora_path = folder_paths.get_full_path_or_raise("loras", matched_lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
                print(f"[Prompt Lora Matcher] Using cached LoRA: {matched_lora_name}")
            else:
                self.loaded_lora = None

        if lora is None:
            print(f"[Prompt Lora Matcher] Loading LoRA from disk: {matched_lora_name}")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        print(f"[Prompt Lora Matcher] Successfully applied LoRA: {matched_lora_name}")
        return (model_lora,)


class AIPoweredLoraMatcher:
    NAME = "AIPoweredLoraMatcher"

    def __init__(self):
        self.loaded_lora = None
        self.classifier = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Text description to match against LoRA descriptions using AI semantic similarity"}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_matched_lora_ai"
    CATEGORY = "loaders"

    def load_matched_lora_ai(self, model, prompt, strength_model):
        print(f"[AI LoRA Matcher] Input prompt: '{prompt}'")

        # Initialize AI classifier if not already done
        if self.classifier is None:
            self.classifier = LLMFolderClassifierAI()

        # Find the best matching lora using AI semantic similarity
        matched_lora_name = self.classifier.classify(prompt)

        if matched_lora_name is None:
            print(f"[AI LoRA Matcher] No matching LoRA found for prompt. Returning original model.")
            return (model,)

        print(f"[AI LoRA Matcher] Matched LoRA: {matched_lora_name}")

        # Load and apply the matched lora
        if strength_model == 0:
            print(f"[AI LoRA Matcher] Strength is 0, returning original model.")
            return (model,)

        print(f"[AI LoRA Matcher] Applying with strength: {strength_model}")

        lora_path = folder_paths.get_full_path_or_raise("loras", matched_lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
                print(f"[AI LoRA Matcher] Using cached LoRA: {matched_lora_name}")
            else:
                self.loaded_lora = None

        if lora is None:
            print(f"[AI LoRA Matcher] Loading LoRA from disk: {matched_lora_name}")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, _ = comfy.sd.load_lora_for_models(model, None, lora, strength_model, 0)
        print(f"[AI LoRA Matcher] Successfully applied LoRA: {matched_lora_name}")
        return (model_lora,)


# Required for ComfyUI to register this node
NODE_CLASS_MAPPINGS = {
    "PromptLoraMatcher": PromptLoraMatcher,
    "AIPoweredLoraMatcher": AIPoweredLoraMatcher
}

# Optional: Human-readable name in the ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptLoraMatcher": "Prompt LoRA Matcher (Fuzzy)",
    "AIPoweredLoraMatcher": "AI LoRA Matcher (Semantic)"
}
