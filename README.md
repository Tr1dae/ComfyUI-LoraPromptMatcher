# LoRA Matcher Nodes for ComfyUI

This custom node provides two different approaches to automatically match text prompts with LoRA models using their descriptions.

## 🎯 Available Nodes

### 1. **Prompt LoRA Matcher (Fuzzy)** - Fast Text Matching
- **Method**: Fuzzy string matching using `rapidfuzz`
- **Speed**: ⚡ Instant matching
- **Accuracy**: Good for exact/near-exact matches
- **Requirements**: None (works out of the box)

### 2. **AI LoRA Matcher (Semantic)** - Smart AI Matching
- **Method**: Semantic similarity using sentence-transformers
- **Speed**: 🐌 Slower first run (downloads model)
- **Accuracy**: 🤖 Understands meaning, synonyms, concepts
- **Requirements**: Sentence-transformer models

## 📁 Model Setup (AI Matcher Only)

The AI matcher can work with or without local models:

### Option 1: Automatic Download (Easiest)
- **No setup required**
- Models download automatically on first use
- Slower initial load (~30-60 seconds)
- Uses `all-MiniLM-L12-v2` by default (higher accuracy)

### Option 2: Local Models (Recommended for Speed)
Place sentence-transformer models in: `custom_nodes/ComfyUI-LoraPromptMatcher/model/`

#### Recommended Models:
1. **all-MiniLM-L12-v2** ⭐⭐ (DEFAULT - Higher accuracy)
   - Size: ~45MB
   - Speed: Medium
   - Accuracy: ⭐⭐⭐⭐⭐ (Best semantic understanding)

2. **all-MiniLM-L6-v2** (Faster alternative)
   - Size: ~23MB
   - Speed: ⚡ Fast
   - Accuracy: ⭐⭐⭐⭐ (Good balance)

3. **paraphrase-MiniLM-L6-v2** (Good for paraphrases)
   - Size: ~23MB
   - Speed: Fast
   - Accuracy: Good for rephrased prompts

#### Download Instructions:
1. Visit: https://huggingface.co/sentence-transformers/
2. Search for your chosen model (e.g., "all-MiniLM-L6-v2")
3. Download the model files
4. Extract to: `custom_nodes/ComfyUI-LoraPromptMatcher/model/`
5. The directory structure should look like:
```
model/
├── all-MiniLM-L6-v2/
│   ├── config_sentence_transformers.json
│   ├── sentence_bert_config.json
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
```

## ⚙️ Configuration

### LoRA Descriptions
Edit `reference_descriptions.json` to add your LoRA models:

```json
{
    "my_lora.safetensors": "beautiful detailed portrait, realistic skin texture",
    "anime_style.safetensors": "anime art style, vibrant colors, expressive eyes",
    "cyberpunk.safetensors": "futuristic cyberpunk aesthetic, neon lights, high tech"
}
```

## 🚨 Warning System

The AI matcher includes intelligent warnings:

### No Models Found
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🤖 AI MODELS NOT FOUND 🤖                          ║
║                                                                            ║
║  The AI LoRA Matcher requires sentence-transformer models for semantic    ║
║  matching, but none were found in the model directory.                    ║
║                                                                            ║
║  RECOMMENDED MODELS:                                                      ║
║  • all-MiniLM-L6-v2 (fast, good balance)                                   ║
║  • all-MiniLM-L12-v2 (better accuracy, slower)                            ║
║  • paraphrase-MiniLM-L6-v2 (good for paraphrases)                         ║
║                                                                            ║
║  DOWNLOAD FROM: huggingface.co/sentence-transformers/                     ║
║  Place model files in: custom_nodes/ComfyUI-LoraPromptMatcher/model/      ║
║                                                                            ║
║  ALTERNATIVE: The node will automatically download models from           ║
║  HuggingFace on first use (may be slower).                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### GGUF Models Found (Wrong Type)
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     🔄 GGUF MODELS FOUND - ST MODELS NEEDED 🔄                ║
║                                                                            ║
║  Found GGUF models but the AI LoRA Matcher needs sentence-transformer     ║
║  models for semantic similarity matching.                                  ║
║                                                                            ║
║  DOWNLOAD sentence-transformer models from:                               ║
║  huggingface.co/sentence-transformers/                                     ║
║                                                                            ║
║  Recommended: all-MiniLM-L6-v2                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## 🎛️ Usage

1. Add either node to your ComfyUI workflow
2. Connect a MODEL input
3. Enter your text description
4. Adjust strength if needed (default: 1.0)
5. The node outputs the model with the matched LoRA applied

## 📊 Matching Examples

| Input Prompt | Fuzzy Match | AI Match | Notes |
|-------------|-------------|----------|--------|
| "beautiful woman portrait" | `portrait_lora` (0.85) | `portrait_lora` (0.92) | Both work well |
| "anime girl cute style" | `anime_lora` (0.76) | `anime_lora` (0.88) | AI better |
| "person eating apple fruit" | `food_lora` (0.34) | `food_lora` (0.78) | AI much better |
| "cyberpunk futuristic city" | No match (0.25) | `cyberpunk_lora` (0.71) | Only AI finds it |

## 🔧 Troubleshooting

### AI Matcher Not Working
- Check console for warning messages
- Ensure model directory exists: `custom_nodes/ComfyUI-LoraPromptMatcher/model/`
- Try automatic download first (delete local models if issues persist)

### No LoRA Applied
- Check `reference_descriptions.json` has your LoRA files
- Verify LoRA files exist in the correct ComfyUI loras folder
- Lower similarity threshold if matches are too strict

### Performance Issues
- Use local models instead of automatic download
- Try smaller models like `all-MiniLM-L6-v2`
- The fuzzy matcher is always faster for simple cases

## 📝 Notes

- The AI matcher caches embeddings for faster subsequent matches
- Fuzzy matching uses token-based similarity (good for typos)
- AI matching uses semantic understanding (better for concepts)
- Both methods have adjustable similarity thresholds
- Console output shows detailed matching information
