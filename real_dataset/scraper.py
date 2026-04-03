import requests
import xml.etree.ElementTree as ET

filename = "real_dataset/pmc_ids.txt"
def get_and_save_pmcids(query, filename, count):
    """
    Searches PMC for a query and saves the IDs to a text file.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": query,
        "retmax": count,
        "retmode": "xml"
    }

    print(f"Searching PMC for: '{query}'...")
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        root = ET.fromstring(response.content)
        ids = [id_tag.text for id_tag in root.findall(".//Id")]
        
        with open(filename, "a") as f:
            for pmcid in ids:
                # E-utilities returns IDs without the 'PMC' prefix, so we add it
                f.write(f"PMC{pmcid}\n")
        
        print(f"Successfully saved {len(ids)} PMC IDs to {filename}")
    else:
        print(f"Failed to connect to NCBI API. Status: {response.status_code}")

# Run the search
# We use 'Kaplan-Meier oncology' to find relevant medical charts
get_and_save_pmcids("Kaplan Meier", filename,count=10000)
get_and_save_pmcids("Kaplan-Meier", filename,count=10000)
get_and_save_pmcids("Kaplan Meier Oncology", filename,count=10000)
get_and_save_pmcids("Kaplan Meier Analysis", filename,count=10000)