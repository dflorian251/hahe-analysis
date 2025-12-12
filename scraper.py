import json
from pathlib import Path
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from abc import ABC, abstractmethod


class Mapper(ABC):
    
    @abstractmethod
    def get_map(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def map(self, key: str) -> str:
        pass

class ProgramMapper(Mapper):

    def __init__(self, map_file: str = "program_map.json") -> None:
        self.__programme_map: Dict[str, str] = json.loads(Path(map_file).read_text())
        
    def get_map(self) -> Dict[str, str]:
        return self.__programme_map
    
    def map(self, program: str) -> str:
        if program.isascii():
            return program
        if self.__programme_map.get(program, "Unknown") == "Unknown":
            print(f"Warning: program '{program}' not found in mapping.")
        return self.__programme_map.get(program, "Unknown")


class InstituteMapper:

    def __init__(self) -> None:
        """
        Initializes the institute_map dictionary as an instance attribute.
        """
        self._institute_map: Dict[str, str] = {
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

    
    def get_map(self) -> Dict[str, str]:
        return self._institute_map
    
    def map(self, institute_id: str) -> str:
        return self._institute_map.get(institute_id, "Unknown Institute")
    
    def get_institutes_ids(self) -> List[str]:
        return list(self._institute_map.keys())

class AcademicYearMapper:

    def __init__(self) -> None:
        """
        Initializes the year_map dictionary as an instance attribute.
        """
        self._year_map: Dict[str, str] = {
            '2022': '2020-2021',
            '2023': '2021-2022',
            '2024': '2022-2023',
            '2025': '2023-2024',
        }

    
    def get_map(self) -> Dict[str, str]:
        """
        Returns the initialized dictionary mapping academic year codes to names.
        """
        return self._year_map
    
    
    def map(self, year: str) -> str:
        """
        Maps a given academic year code to its standardized name.
        
        Args:
            year (str): The academic year code to map.
        
        Returns:
            str: The standardized academic year name or "Unknown academic year" if not found.
        """
        return self._year_map.get(year, "Unknown academic year")


def save_to_csv(data: List[List[Dict[str, Any]]], filename: str) -> None:
    """Save the list of dictionaries to a CSV file."""
    flat_data: List[Dict[str, Any]] = [item for sublist in data for item in sublist]
    df = pd.DataFrame(flat_data)
    df.to_csv(filename, index=False)


def scrape_data(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Scrape data from the BeautifulSoup object and return a list of dictionaries."""
    soup_data = soup.find_all('div', class_='stats-item')   # get all the programmes
    scraped_data: List[Dict[str, Any]] = []
    for item in soup_data:
        program = item.find('div', class_='stats-item-title').text.strip()
        program = re.search("^ΠΠΣ ::", program) or re.search("^ΠΠΣ ΕΑΠ ::", program) or re.search("^ΞΠΣ ::", program)

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

        program = program.string[program.end():].strip() if program else "Unknown"
        program = re.sub(r"[\(\[].*?[\)\]]", "", program).strip()
        program = program.replace(" ΤΕ", "").replace('"', "").strip()
        scraped_data.append({
            "institution": InstituteMapper().get_map().get(data.get("filter[instituteid]")) if data.get("filter[instituteid]") else "Unknown",
            "academic_year": AcademicYearMapper().map(data.get("filter[collectionyear]")),
            "program": ProgramMapper().map(program.strip('"')),
            "established": established.string[established.end():] if established else "Unknown",
            "graduate": int(graduate.replace('.', '')) if graduate != '--' else 0,
            "registered": int(registered.replace('.', '')) if registered != '--' else 0,
            "enrolled": int(enrolled.replace('.', '')) if enrolled != '--' else 0,
            "active": int(active.replace('.', '')) if active != '--' else 0,
        })
    return scraped_data

URL = "https://www.ethaae.gr/el/dedomena-aei/foitites-aei"

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Origin": "https://www.ethaae.gr",
    "Referer": "https://www.ethaae.gr/el/dedomena-aei/foitites-aei",
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


collected_data = []
institude_mappper = InstituteMapper()
for year in range(2022, 2026):    
    for inst_id in InstituteMapper().get_institutes_ids():
        data["filter[collectionyear]"] = str(year)
        data["filter[instituteid]"] = str(inst_id)

        response = requests.post(URL, headers=HEADERS, data=data)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        collected_data.append(scrape_data(soup))

save_to_csv(collected_data, f"hahe_all_21_24.csv")
