import torch
from transformers import AutoModel, AutoTokenizer
import os
import csv
import json
import re
from PIL import Image

#Constants
MODEL = "deepseek-ai/DeepSeek-OCR"
DATA_FILE = "../KIE-HVQA/kie_hocr.jsonl"  # Path to the test data
IMAGE_BASE_PATH = "../KIE-HVQA"  # Base path for images

#Functions
def save_result_incrementally_csv(result, file_path):
    """
    Save results
    """
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

def save_result_incrementally_jsonl(result, file_path):
    """
    Save results to JSONL
    """
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(result, ensure_ascii=False) + '\n')

def parse_degradation_tags(text):
    """
    Parse text with degradation tags and separate into:
    - clear characters (fully legible)
    - not clear characters (partially occluded or completely occluded)
    - final OCR (complete text)
    """
    clear_chars = ""
    not_clear_chars = ""
    final_ocr = ""
    
    temp = text
    
    #Find all <part_occluded>[X] patterns (partially visible characters)
    part_occluded_pattern = r'<part_occluded>\[(.?)\]'
    part_occluded_matches = re.findall(part_occluded_pattern, temp)
    
    #Find all <occluded> patterns (completely hidden characters)
    occluded_pattern = r'<occluded>'
    occluded_count = len(re.findall(occluded_pattern, temp))
    
    #Extract clear characters (everything not in tags)
    clear_text = re.sub(r'<part_occluded>\[.?\]', '', temp)
    clear_text = re.sub(r'<occluded>', '', clear_text)
    clear_chars = clear_text
    
    #Not clear characters = partially occluded ones
    not_clear_chars = ''.join(part_occluded_matches)
    
    #Alternative: reconstruct final by just removing tags
    final_ocr_clean = temp
    final_ocr_clean = re.sub(r'<part_occluded>\[(.?)\]', r'\1', final_ocr_clean)
    final_ocr_clean = re.sub(r'<occluded>', '', final_ocr_clean)
    
    return {
        "clear Char-level OCR": clear_chars,
        "not clear enough Char-level OCR": not_clear_chars,
        "Final OCR": final_ocr_clean
    }

# System check
print("="*50)
print("SYSTEM CHECK")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*50)

#Setup device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = "cuda" if torch.cuda.is_available() else "cpu"

#Load model with memory optimization
print(f"\nLoading {MODEL}...")
print("Using memory-optimized settings for 6GB GPU...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print("✓ Tokenizer loaded")
    
    # Load model with smart memory management
    # Try with torch_dtype set to "auto" to let the model decide
    model = AutoModel.from_pretrained(
        MODEL, 
        trust_remote_code=True,
        device_map="auto",  # Automatic device placement
        torch_dtype="auto",  # Let model use its preferred dtype
        low_cpu_mem_usage=True,  # Minimize memory usage during loading
    )
    print("✓ Model loaded with automatic device mapping")
    print(f"  Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'single device'}")
    
    model = model.eval()
    print("✓ Model ready for inference")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise

#Load KIE-HVQA dataset from local file
print(f"\nLoading KIE-HVQA dataset from {DATA_FILE}...")
dataset = []
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    print(f"✓ Dataset loaded: {len(dataset)} examples")
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    raise

output_csv = "kie_hvqa_deepseek_ocr_results.csv"
output_jsonl = "kie_hvqa_deepseek_ocr_results.jsonl"
output_dir = "./outputs/kie_hvqa"
os.makedirs(output_dir, exist_ok=True)

#Clear previous JSONL file if exists
if os.path.exists(output_jsonl):
    os.remove(output_jsonl)

print(f"\n{'='*50}")
print(f"Starting evaluation on {len(dataset)} examples...")
print(f"{'='*50}\n")

for idx, row in enumerate(dataset):
    print(f"\n[{idx + 1}/{len(dataset)}] Processing example {idx} (ID: {row.get('id', 'unknown')})...")
    
    # Get image path - make it absolute
    image_path = row['image']
    # Fix path: change 'images' to 'data' since that's where files actually are
    image_path = image_path.replace('./images/', './data/')
    if not os.path.isabs(image_path):
        # Image path is relative, combine with base path
        image_path = os.path.join(IMAGE_BASE_PATH, image_path)
    
    # Normalize path separators for Windows
    image_path = os.path.normpath(image_path)
    
    if not os.path.exists(image_path):
        print(f"  ⚠ Warning: Image not found at {image_path}")
        continue
    
    # Extract question from problem (remove <image> tag)
    problem = row['problem']
    question = problem.replace("<image>", "").replace("\n", " ").strip()
    
    # Parse ground truth
    ground_truth = row['answer']
    
    print(f"  Question: {question[:80]}...")
    
    # The ground truth is already in JSON format
    try:
        gt_parsed = json.loads(ground_truth)
    except:
        # If it's not valid JSON, parse it as before
        gt_parsed = parse_degradation_tags(ground_truth)
    
    prompt = f"<image>\n<|grounding|>{question}"
    
    try:
        print("  Running inference...")
        res = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_path,
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            test_compress=True
        )
        
        generated_text = res if isinstance(res, str) else str(res)
        print(f"  ✓ Generated: {generated_text[:100]}...")
        
        # Try to parse response as JSON if it looks like JSON
        try:
            pred_parsed = json.loads(generated_text)
            # Ensure it has the required keys
            if "clear Char-level OCR" not in pred_parsed:
                pred_parsed = {
                    "clear Char-level OCR": generated_text,
                    "not clear enough Char-level OCR": "",
                    "Final OCR": generated_text
                }
        except:
            pred_parsed = {
                "clear Char-level OCR": generated_text,
                "not clear enough Char-level OCR": "",
                "Final OCR": generated_text
            }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        generated_text = f"[ERROR: {str(e)}]"
        pred_parsed = {
            "clear Char-level OCR": "",
            "not clear enough Char-level OCR": "",
            "Final OCR": generated_text
        }
    
    #Save to CSV 
    csv_result = {
        "idx": idx,
        "id": row.get('id', 'unknown'),
        "image_path": image_path,
        "question": question,
        "ground_truth_raw": ground_truth,
        "ground_truth_clear": gt_parsed.get("clear Char-level OCR", ""),
        "ground_truth_notclear": gt_parsed.get("not clear enough Char-level OCR", ""),
        "ground_truth_final": gt_parsed.get("Final OCR", ""),
        "generated_text": generated_text,
    }
    save_result_incrementally_csv(csv_result, output_csv)
    
    #Save to JSONL (format for eval.py)
    jsonl_result = {
        "answer": ground_truth,  # Keep original format
        "response": json.dumps(pred_parsed, ensure_ascii=False)
    }
    save_result_incrementally_jsonl(jsonl_result, output_jsonl)
    
    # Clear CUDA cache periodically to prevent memory buildup
    if torch.cuda.is_available() and idx % 10 == 0:
        torch.cuda.empty_cache()

print(f"\n{'='*50}")
print(f"Evaluation complete!")
print(f"{'='*50}")
print(f"CSV results: {output_csv}")
print(f"JSONL results: {output_jsonl}")
print(f"Output directory: {output_dir}")
print("\nTo calculate benchmark scores:")
print("  cd ../KIE-HVQA")
print(f"  python eval.py --results_file ../src/{output_jsonl}")