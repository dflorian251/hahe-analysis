import requests
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
import os
import re


def save_to_csv(data: list[dict], filename: str):
    """Save the list of dictionaries to a CSV file.
    
    Args:
        data (list): List of dictionaries containing the scraped data.
        filename (str): The name of the CSV file to save the data.
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def scrape_data(soup: BeautifulSoup):
    """Scrape data from the BeautifulSoup object and return a list of dictionaries.
    
    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the HTML content.
    """
    data = []
    soup_data = soup.find_all('div', class_='stats-item')
    for item in soup_data:
        institution = item  .find('div', class_='stats-item-title').text.strip()
        institution = re.search("^ΙΔΡΥΜΑ ::", institution)
        
        established = item.find('span', class_='stats-item-date').text.strip()
        established = re.search("Ημ/νία Ίδρυσης: ", established)
        
        variables = item.find_all('div', class_='stats-item-variable-title')
        print(f"variables: {variables}")
        for var in variables:
            print(var)
            variable = var.text.strip()
            if variable == "Απόφοιτοι ΠΠΣ Ιδρύματος":
                graduates = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()
            elif variable == "Εγγεγραμμένοι φοιτητές ΠΠΣ Ιδρύματος":
                registed = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()
            elif variable == "Εισαχθέντες φοιτητές ΠΠΣ Ιδρύματος":
                freshmen = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()
            elif variable == "Ενεργοί φοιτητές ΠΠΣ Ιδρύματος":
                active = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()

        data.append({
            'institution': institution.string[institution.end():].strip(),
            'established': established.string[established.end():],
            "graduates": graduates,
            "registed": registed,
            "freshmen": freshmen,
            "active": active,
        })

        save_to_csv(data, f'hahe_{payload.get("filter[collectionyear]")}.csv')


load_dotenv()
DATA_SOURCE = os.getenv("DATA_SOURCE")

payload = {
    "filter[collectionyear]": "2025",   # 2023-2024
    "filter[instituteid]": "",  # All institutes
    "list[limit]": 0,   # All records
}

for year in range(2020, 2026):
    payload["filter[collectionyear]"] = str(year)

    response = requests.post(DATA_SOURCE, data=payload)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    scrape_data(soup)
