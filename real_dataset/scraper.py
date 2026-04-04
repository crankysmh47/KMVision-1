import requests
import xml.etree.ElementTree as ET
import os
import time

def get_and_save_plos_ids(query, filename, count=500):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Base URL for NCBI E-Search
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    # REFACTORED QUERY: 
    # Using 'PLOS ONE'[Journal] is the most reliable way to get results.
    # We use quotes around the query to handle spaces.
    search_term = f'"{query}" AND "PLOS ONE"[Journal]'
    
    params = {
        "db": "pmc",
        "term": search_term,
        "retmax": count,
        "retmode": "xml",
        "tool": "MedicalDataCollector",
        "email": "your_email@example.com" # NCBI likes having an email for large requests
    }

    print(f"🚀 Searching PMC for: {search_term}")
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            
            # Debug: Print the count NCBI actually found
            found_count = root.find("Count").text if root.find("Count") is not None else "0"
            print(f"NCBI found {found_count} total matches.")
            
            ids = [id_tag.text for id_tag in root.findall(".//Id")]
            
            if not ids:
                print(f"Zero IDs returned in the XML for '{query}'.")
                return 0

            with open(filename, "a", encoding="utf-8") as f:
                for pmcid in ids:
                    f.write(f"PMC{pmcid}\n")
            
            print(f"Successfully added {len(ids)} IDs to {filename}")
            return len(ids)
        else:
            print(f"❌ API Error: {response.status_code}")
            return 0
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return 0

# --- Execution ---
filename1 = "real_dataset/plos_id_wf.txt"
filename2 = "real_dataset/plos_id_km.txt"
filename3 = "real_dataset/plos_id_forest.txt"

get_and_save_plos_ids("Kaplan-Meier Survival", filename2, count=1000)
time.sleep(1) 
get_and_save_plos_ids("Survival Analysis Kaplan Meier", filename2, count=1000)
time.sleep(1) 
get_and_save_plos_ids("Forest Plot", filename3, count=1000)
time.sleep(1) 
get_and_save_plos_ids("Forest Plot meta analysis", filename3, count=1000)
time.sleep(1) 
get_and_save_plos_ids("Waterfall Plot", filename1, count=1000)
time.sleep(1) 
get_and_save_plos_ids("Waterfall plot oncology", filename1, count=1000)