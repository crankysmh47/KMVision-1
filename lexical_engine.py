import os
import json
import random

DATA_DIR = r"C:\sem4\KMVision-1 Data\config"
MEDICAL_FILE = os.path.join(DATA_DIR, "medical_corpus.json")
ENG_FILE = os.path.join(DATA_DIR, "english_dictionary.json")

med_words = ["Control", "Treatment", "Survival", "Response", "Progression"]
eng_words = ["apple", "dog", "house", "car", "banana"]

try:
    with open(MEDICAL_FILE, 'r') as f:
        med_words = json.load(f)
except:
    pass

try:
    with open(ENG_FILE, 'r') as f:
        eng_words = json.load(f)
except:
    pass

MEDICAL_PREFIXES = ["Estimated", "Adjusted", "Mean", "Median", "Baseline", "Post", "Aggregate", "Overall", "Comparative"]
MEDICAL_METRICS = med_words
MEDICAL_UNITS = ["(%)", "(Months)", "(Days)", "(Years)", "(mg/mL)", "(ng/dL)", "Index", "Ratio", "Score", "/ week"]

def apply_typo_noise(text: str) -> str:
    # 30% chance for typographic noise to simulate bad scans/OCR
    if random.random() > 0.3:
        return text
        
    noise_type = random.choice(["casing", "drops", "spacing", "all"])
    
    if noise_type in ["casing", "all"]:
        case_op = random.choice(["upper", "lower", "mixed"])
        if case_op == "upper": text = text.upper()
        elif case_op == "lower": text = text.lower()
        else: text = "".join([c.upper() if random.random() > 0.5 else c.lower() for c in text])
            
    if noise_type in ["spacing", "all"]:
        # Increase intra-word spacing
        text = text.replace(" ", "  ") if random.random() < 0.5 else "   ".join(text.split(" "))
        
    if noise_type in ["drops", "all"]:
        # Drop random chars to simulate OCR dropping
        if len(text) > 4:
            drop_idx = random.randint(1, len(text)-2)
            text = text[:drop_idx] + text[drop_idx+1:]
                
    return text

def generate_label() -> str:
    """
    Generates text labels matching strict 80/20 distribution:
    80%: Clinical Combinatorics (equal odds 1, 2, or 3 words)
    20%: Dictionary Anarchy
    """
    if random.random() < 0.20:
        # Dictionary Anarchy
        num_words = random.randint(1, 3)
        words = [random.choice(eng_words).capitalize() for _ in range(num_words)]
        label = " ".join(words)
    else:
        # Clinical Combinatorics
        num_words = random.choice([1, 2, 3])
        if num_words == 1:
            label = random.choice(MEDICAL_METRICS)
        elif num_words == 2:
            if random.random() < 0.5:
                label = f"{random.choice(MEDICAL_PREFIXES)} {random.choice(MEDICAL_METRICS)}"
            else:
                label = f"{random.choice(MEDICAL_METRICS)} {random.choice(MEDICAL_UNITS)}"
        else:
            label = f"{random.choice(MEDICAL_PREFIXES)} {random.choice(MEDICAL_METRICS)} {random.choice(MEDICAL_UNITS)}"
            
    return apply_typo_noise(label)
