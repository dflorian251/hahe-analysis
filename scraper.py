import requests
from bs4 import BeautifulSoup
import pandas as pd
from dotenv import load_dotenv
import os
import re


load_dotenv()
DATA_SOURCE = os.getenv("DATA_SOURCE")


response = requests.get(DATA_SOURCE)
response.raise_for_status()

soup = BeautifulSoup(response.content, "html.parser")

data = []
soup_data = soup.find_all('div', class_='stats-item')
for item in soup_data:
    institution = item.find('div', class_='stats-item-title').text.strip()
    institution = re.search("^ΙΔΡΥΜΑ ::", institution)
    # print(f"institution: {institution.string[institution.end():]}")
    established = item.find('span', class_='stats-item-date').text.strip()
    established = re.search("Ημ/νία Ίδρυσης: ", established)
    # print(f"established: {established.string[established.end():]}")

    data.append({
        'institution': institution.string[institution.end():],
        'established': established.string[established.end():]
    })

df = pd.DataFrame(data)
df.to_csv('hahe.csv', index=False)


