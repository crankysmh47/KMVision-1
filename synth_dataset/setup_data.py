import os
import urllib.request
import json
import random

DATA_DIR = r"C:\sem4\KMVision-1 Data\config"
os.makedirs(DATA_DIR, exist_ok=True)

MED_DICT_URL = "https://raw.githubusercontent.com/glutanimate/wordlist-medicalterms-en/master/wordlist.txt"
ENG_DICT_URL = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"

MEDICAL_WORDS_FILE = os.path.join(DATA_DIR, "medical_corpus.json")
ENGLISH_WORDS_FILE = os.path.join(DATA_DIR, "english_dictionary.json")

def download_and_process():
    print("Fetching Medical corpus...")
    req = urllib.request.Request(MED_DICT_URL, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            med_words = response.read().decode('utf-8').splitlines()
            # The list might be large, we want the most frequent/relevant.
            # Filtering for words with length 5-12, purely alphabetical to avoid bizarre junk.
            med_words = [w.strip().capitalize() for w in med_words if w.isalpha() and 5 <= len(w) <= 12]
            random.seed(42) # Deterministic sample of 2k
            med_sample = random.sample(med_words, min(2000, len(med_words)))
            
            with open(MEDICAL_WORDS_FILE, 'w') as f:
                json.dump(med_sample, f, indent=2)
            print(f"Saved {len(med_sample)} medical words.")
    except Exception as e:
        print("Failed to download medical words:", e)

    print("Fetching Standard English Dictionary...")
    req_eng = urllib.request.Request(ENG_DICT_URL, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req_eng) as response:
            eng_words = response.read().decode('utf-8').splitlines()
            # Filter for common sized dictionary words
            eng_words = [w.strip().lower() for w in eng_words if w.isalpha() and 3 <= len(w) <= 12]
            with open(ENGLISH_WORDS_FILE, 'w') as f:
                json.dump(eng_words, f)
            print(f"Saved {len(eng_words)} standard english words.")
    except Exception as e:
        print("Failed to download english words:", e)

if __name__ == '__main__':
    download_and_process()
