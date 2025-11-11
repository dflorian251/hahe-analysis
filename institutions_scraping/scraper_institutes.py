import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

url = "https://www.ethaae.gr/el/dedomena-aei/foitites-aei"

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.ethaae.gr",
    "Referer": "https://www.ethaae.gr/el/dedomena-aei/foitites-aei",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Sec-GPC": "1",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Chromium";v="142", "Brave";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Linux"',
}

cookies = {
    "743132fc25826b25d8637ec61f9a7943": "67801750e61a1f566d403a78a0e90b5a",
}

data = {
    "filter[collectionyear]": "2025",
    "filter[instituteid]": "4",
    "list[limit]": "0",
    "task": "",
    "boxchecked": "0",
    "filter_order": "",
    "filter_order_Dir": "",
    "limitstart": "",
    "4703fa291a36682ac6cea0d1cd8f4da8": "1",
    "list[fullordering]": "null ASC",
}

def save_to_csv(data: list[list[dict]], filename: str):
    """Save the list of dictionaries to a CSV file.
    
    Args:
        data (list): List of dictionaries containing the scraped data.
        filename (str): The name of the CSV file to save the data.
    """
    flat_data = [item for sublist in data for item in sublist]
    df = pd.DataFrame(flat_data)
    df.to_csv(filename, index=False)


def map_academic_year(year: str) -> str:
    """Map academic year codes to standardized names."""
    year_map = {
        '2022': '2020-2021',
        '2023': '2021-2022',
        '2024': '2022-2023',
        '2025': '2023-2024',
    }
    return year_map.get(year, "Unknown academic year")


class InstituteMapper:
    """
    A utility class to map institution IDs to their English names for Greek 
    universities and academies, initializing the map upon object creation.
    """
    def __init__(self):
        """
        Initializes the institute_map dictionary as an instance attribute.
        """
        self._institute_map = {
            # '': '- Select Institute -', 
            '1': 'Athens School of Fine Arts',
            '2': 'Aristotle University of Thessaloniki',
            '36': 'School of Pedagogical and Technological Education (ASPETE)',
            '3': 'Agricultural University of Athens',
            '4': 'Democritus University of Thrace',
            '131': 'International Hellenic University',
            '5': 'National and Kapodistrian University of Athens',
            '6': 'National Technical University of Athens',
            '22': 'Hellenic Open University',
            '130': 'Hellenic Mediterranean University',
            '7': 'Ionian University',
            '8': 'Athens University of Economics and Business',
            '9': 'University of the Aegean',
            '129': 'University of West Attica',
            '10': 'University of Western Macedonia',
            '11': 'University of Thessaly',
            '12': 'University of Ioannina',
            '13': 'University of Crete',
            '14': 'University of Macedonia',
            '15': 'University of Patras',
            '16': 'University of Piraeus',
            '17': 'University of Peloponnese',
            '18': 'Panteion University of Social and Political Sciences',
            '19': 'Technical University of Crete',
            '39': 'Hellenic Army Academy',
            '40': 'Hellenic Air Force Academy',
            '41': 'Hellenic Naval Academy',
            '20': 'Harokopio University'
        }

    def get_institute_map(self):
        """
        Returns the initialized dictionary mapping institution IDs to English names.
        """
        return self._institute_map
    
    def get_institutes_ids(self):
        """
        Returns a list of all institution IDs in the mapping.
        """
        return list(self._institute_map.keys())


def scrape_data(soup: BeautifulSoup):
    """Scrape data from the BeautifulSoup object and return a list of dictionaries.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the HTML content.
    """
    institute_mapper = InstituteMapper()
    soup_data = soup.find_all('div', class_='stats-item')   # get all the programmes
    scraped_data = []
    for item in soup_data:
        programme = item.find('div', class_='stats-item-title').text.strip()
        programme = re.search("^ΠΠΣ ::", programme)

        established = item.find('span', class_='stats-item-date').text.strip()
        established = re.search("^Ημ/νία Ίδρυσης: ", established)

        variables = item.find_all('div', class_='stats-item-variable-title')
        for var in variables:
            variable = var.text.strip()
            if variable == "Απόφοιτοι ΠΠΣ":
                graduate = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()
            elif variable == "Εγγεγραμμένοι φοιτητές ΠΠΣ":
                registered = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()
            elif variable == "Εισαχθέντες φοιτητές ΠΠΣ":
                enrolled = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()
            elif variable == "Ενεργοί φοιτητές ΠΠΣ":
                active = var.find_next_sibling('div', class_='stats-item-variable-value').text.strip()

        programme = programme.string[programme.end():].strip() if programme else "Unknown"
        scraped_data.append({
            "institution": institute_mapper.get_institute_map().get(data.get("filter[instituteid]")) if data.get("filter[instituteid]") else "Unknown",
            "academic_year": map_academic_year(data.get("filter[collectionyear]")),
            "programme": programme.strip('"'),
            'established': established.string[established.end():] if established else "Unknown",
            "graduate": int(graduate.replace('.', '')) if graduate != '--' else 0,
            "registered": int(registered.replace('.', '')) if registered != '--' else 0,
            "enrolled": int(enrolled.replace('.', '')) if enrolled != '--' else 0,
            "active": int(active.replace('.', '')) if active != '--' else 0,
        })
    return scraped_data

collected_data = []
print(InstituteMapper().get_institutes_ids())
for year in range(2022, 2026):    
    for inst_id in InstituteMapper().get_institutes_ids():
        data["filter[collectionyear]"] = str(year)
        data["filter[instituteid]"] = str(inst_id)

        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        collected_data.append(scrape_data(soup))
    print(collected_data)

save_to_csv(collected_data, f"hahe_all_institutes_2022_2025.csv")
