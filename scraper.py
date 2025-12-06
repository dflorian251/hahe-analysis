from typing import List, Dict
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from abc import ABC, abstractmethod


class Mapper(ABC):
    
    @abstractmethod
    def get_map(self) -> dict:
        pass

    @abstractmethod
    def map(self, key: str) -> str:
        pass


class ProgramMapper(Mapper):

    def __init__(self):
        self.__programme_map = {
            'Εικαστικών Τεχνών': 'Fine Arts',
            'Θεωρίας και Ιστορίας της Τέχνης': 'Art Theory and History',
            'Αγγλικής Γλώσσας και Φιλολογίας': 'English Language and Literature',
            'Αγρονόμων και Τοπογράφων Μηχανικών': 'Agricultural and Surveying Engineering',
            'Αρχιτεκτόνων Μηχανικών': 'Architectural Engineering',
            'Βιολογίας': 'Biology',
            'Γαλλικής Γλώσσας και Φιλολογίας': 'French Language and Literature',
            'Γερμανικής Γλώσσας και Φιλολογίας': 'German Language and Literature',
            'Γεωλογίας': 'Geology',
            'Γεωπονίας': 'Agriculture',
            'Δασολογίας και Φυσικού Περιβάλλοντος': 'Forestry and Natural Environment',
            'Δημοσιογραφίας και Μέσων Μαζικής Επικοινωνίας': 'Journalism and Mass Media',
            'Εικαστικών και Εφαρμοσμένων Τεχνών': 'Fine and Applied Arts',
            'Εισαγωγική κατεύθυνση Μουσουλμανικών Σπουδών': 'Introductory Direction of Muslim Studies',
            'Επιστήμης Φυσικής Αγωγής και Αθλητισμού': 'Physical Education and Sports Science',
            'Επιστημών Προσχολικής Αγωγής και Εκπαίδευσης': 'Preschool Education Sciences',
            'Ηλεκτρολόγων Μηχανικών και Μηχανικών Υπολογιστών': 'Electrical and Computer Engineering',
            'Θεάτρου': 'Theater',
            'Θεολογίας': 'Theology',
            'Ιατρικής': 'Medicine',
            'Ιστορίας και Αρχαιολογίας': 'History and Archaeology',
            'Ιταλικής Γλώσσας και Φιλολογίας': 'Italian Language and Literature',
            'Κινηματογράφου': 'Cinema',
            'Κοινωνικής Θεολογίας και Χριστιανικού Πολιτισμού': 'Social Theology and Christian Culture',
            'Κτηνιατρικής': 'Veterinary Medicine',
            'Μαθηματικών': 'Mathematics',
            'Μηχανικών Χωροταξίας και Ανάπτυξης': 'Spatial Planning and Development Engineering',
            'Μηχανολόγων Μηχανικών': 'Mechanical Engineering',
            'Μουσικών Σπουδών': 'Music Studies',
            'Νομικής': 'Law',
            'Οδοντιατρικής': 'Dentistry',
            'Οικονομικών Επιστημών': 'Economic Sciences',
            'Παιδαγωγικό Δημοτικής Εκπαίδευσης': 'Primary Education Pedagogy',
            'Πληροφορικής': 'Informatics',
            'Πολιτικών Επιστημών': 'Political Sciences',
            'Πολιτικών Μηχανικών': 'Civil Engineering',
            'Φαρμακευτικής': 'Pharmacy',
            'Φιλολογίας': 'Philology',
            'Φιλοσοφίας και Παιδαγωγικής': 'Philosophy and Pedagogy',
            'Φυσικής': 'Physics',
            'Χημείας': 'Chemistry',
            'Χημικών Μηχανικών': 'Chemical Engineering',
            'Ψυχολογίας': 'Psychology',
            'Εκπαιδευτικών Ηλεκτρολόγων Μηχανικών': 'Educational Electrical Engineering',
            'Εκπαιδευτικών Ηλεκτρονικών Μηχανικών': 'Educational Electronic Engineering',
            'Εκπαιδευτικών Μηχανολόγων Μηχανικών': 'Educational Mechanical Engineering',
            'Εκπαιδευτικών Πολιτικών Μηχανικών': 'Educational Civil Engineering',
            'Αγροτικής Οικονομίας και Ανάπτυξης': 'Agricultural Economy and Development',
            'Αξιοποίησης Φυσικών Πόρων και Γεωργικής Μηχανικής': 'Natural Resources Utilization and Agricultural Engineering',
            'Βιοτεχνολογίας': 'Biotechnology',
            'Δασολογίας και Διαχείρισης Φυσικού Περιβάλλοντος': 'Forestry and Natural Environment Management',
            'Διοίκησης Γεωργικών Επιχειρήσεων και Συστημάτων Εφοδιασμού': 'Agricultural Business Management and Supply Systems',
            'Επιστήμης Ζωικής Παραγωγής': 'Animal Production Science',
            'Επιστήμης Τροφίμων και Διατροφής του Ανθρώπου': 'Food Science and Human Nutrition',
            'Επιστήμης Φυτικής Παραγωγής': 'Plant Production Science',
            'Περιφερειακής και Οικονομικής Ανάπτυξης': 'Regional and Economic Development',
            'Δασοπονίας και Διαχείρισης Φυσικού Περιβάλλοντος': 'Forestry and Natural Environment Management',
            'Διοίκησης Συστημάτων Εφοδιασμού': 'Supply Systems Management',
            'Διοίκησης, Οικονομίας και Επικοινωνίας Πολιτιστικών και Τουριστικών Μονάδων': 'Management, Economy and Communication of Cultural and Tourist Units',
            'Αγροτικής Ανάπτυξης': 'Agricultural Development',
            'Γλώσσας, Φιλολογίας και Πολιτισμού Παρευξεινίων Χωρών': 'Language, Philology and Culture of Black Sea Countries',
            'Δασολογίας και Διαχείρισης Περιβάλλοντος και Φυσικών Πόρων': 'Forestry and Environment and Natural Resources Management',
            'Ελληνικής Φιλολογίας': 'Greek Philology',
            'Επιστημών Εκπαίδευσης στην Προσχολική Ηλικία': 'Preschool Education Sciences',
            'Ιστορίας και Εθνολογίας': 'History and Ethnology',
            'Κοινωνικής Εργασίας': 'Social Work',
            'Κοινωνικής Πολιτικής': 'Social Policy',
            'Μηχανικών Παραγωγής και Διοίκησης': 'Production and Management Engineering',
            'Μηχανικών Περιβάλλοντος': 'Environmental Engineering',
            'Μοριακής Βιολογίας και Γενετικής': 'Molecular Biology and Genetics',
            'Πολιτικής Επιστήμης': 'Political Science',
            'Αγροτικής Βιοτεχνολογίας και Οινολογίας': 'Agricultural Biotechnology and Oenology',
            'Αγωγής και Φροντίδας στην Πρώιμη Παιδική Ηλικία': 'Early Childhood Care and Education',
            'Βιβλιοθηκονομίας, Αρχειονομίας και Συστημάτων Πληροφόρησης': 'Library Science, Archival Science and Information Systems',
            'Βιοϊατρικών Επιστημών': 'Biomedical Sciences',
            'Δημιουργικού Σχεδιασμού και Ένδυσης': 'Creative Design and Clothing',
            'Διοίκησης Εφοδιαστικής Αλυσίδας': 'Supply Chain Management',
            'Διοίκησης Οργανισμών, Μάρκετινγκ και Τουρισμού': 'Organization Management, Marketing and Tourism',
            'Διοικητικής Επιστήμης και Τεχνολογίας': 'Management Science and Technology',
            'Επιστήμης και Τεχνολογίας Τροφίμων': 'Food Science and Technology',
            'Επιστημών Διατροφής και Διαιτολογίας': 'Nutrition and Dietetics Sciences',
            'Εσωτερικής Αρχιτεκτονικής': 'Interior Architecture',
            'Λογιστικής και Πληροφοριακών Συστημάτων': 'Accounting and Information Systems',
            'Λογιστικής και Χρηματοοικονομικής': 'Accounting and Finance',
            'Μαιευτικής': 'Midwifery',
            'Μηχανικών Πληροφορικής και Ηλεκτρονικών Συστημάτων': 'Informatics and Electronic Systems Engineering',
            'Μηχανικών Πληροφορικής, Υπολογιστών και Τηλεπικοινωνιών': 'Informatics, Computers and Telecommunications Engineering',
            'Μηχανικών Τοπογραφίας και Γεωπληροφορικής': 'Surveying and Geoinformatics Engineering',
            'Νοσηλευτικής': 'Nursing',
            'Οργάνωσης και Διοίκησης Επιχειρήσεων': 'Business Administration',
            'Φυσικοθεραπείας': 'Physiotherapy',
            'Αισθητικής και Κοσμητολογίας': 'Aesthetics and Cosmetology',
            'Αρχιτεκτονικής Τοπίου': 'Landscape Architecture',
            'Βιβλιοθηκονομίας και Συστημάτων Πληροφόρησης': 'Library Science and Information Systems',
            'Διατροφής και Διαιτολογίας': 'Nutrition and Dietetics',
            'Διοίκησης Επιχειρήσεων': 'Business Administration',
            'Διοίκησης Τουριστικών Επιχειρήσεων και Επιχειρήσεων Φιλοξενίας': 'Tourism Business and Hospitality Management',
            'Εσωτερικής Αρχιτεκτονικής, Διακόσμησης και Σχεδιασμού Αντικειμένων': 'Interior Architecture, Decoration and Object Design',
            'Ηλεκτρολόγων Μηχανικών': 'Electrical Engineering',
            'Ηλεκτρονικών Μηχανικών': 'Electronic Engineering',
            'Ιατρικών Εργαστηρίων': 'Medical Laboratories',
            'Μηχανικών Αυτοματισμού': 'Automation Engineering',
            'Μηχανικών Πληροφορικής': 'Informatics Engineering',
            'Μηχανικών Τεχνολογίας Πετρελαίου και Φυσικού Αερίου': 'Oil and Natural Gas Technology Engineering',
            'Μηχανολόγων Οχημάτων': 'Vehicle Mechanical Engineering',
            'Οινολογίας και Τεχνολογίας Ποτών': 'Oenology and Beverage Technology',
            'Προσχολικής Αγωγής': 'Preschool Education',
            'Σχεδιασμού και Τεχνολογίας Ένδυσης': 'Clothing Design and Technology',
            'Τεχνολογίας Τροφίμων': 'Food Technology',
            'Τεχνολόγων Γεωπόνων': 'Agricultural Technologists',
            'Unknown': 'Unknown',
            'Αγροτικής Ανάπτυξης, Αγροδιατροφής και Διαχείρισης Φυσικών Πόρων': 'Agricultural Development, Agrifood and Natural Resources Management',
            'Αεροδιαστημικής Επιστήμης και Τεχνολογίας': 'Aerospace Science and Technology',
            'Γεωλογίας και Γεωπεριβάλλοντος': 'Geology and Geoenvironment',
            'Διαχείρισης Λιμένων και Ναυτιλίας': 'Port and Shipping Management',
            'Διοίκησης Επιχειρήσεων και Οργανισμών': 'Business Administration',
            'Εκπαίδευσης και Αγωγής στην Προσχολική Ηλικία': 'Preschool Education and Training',
            'Επικοινωνίας και Μέσων Μαζικής Ενημέρωσης': 'Communication and Mass Media',
            'Θεατρικών Σπουδών': 'Theater Studies',
            'Ισπανικής Γλώσσας και Φιλολογίας': 'Spanish Language and Literature',
            'Ιστορίας και Φιλοσοφίας της Επιστήμης': 'History and Philosophy of Science',
            'Κοινωνικής Θεολογίας και Θρησκειολογίας': 'Social Theology and Religious Studies',
            'Κοινωνιολογίας': 'Sociology',
            'Παιδαγωγικό Τμήμα Δευτεροβάθμιας Εκπαίδευσης': 'Secondary Education Pedagogy Department',
            'Πληροφορικής και Τηλεπικοινωνιών': 'Informatics and Telecommunications',
            'Πολιτικής Επιστήμης και Δημόσιας Διοίκησης': 'Political Science and Public Administration',
            'Ρωσικής Γλώσσας και Φιλολογίας και Σλαβικών Σπουδών': 'Russian Language and Literature and Slavic Studies',
            'Τεχνολογιών Ψηφιακής Βιομηχανίας': 'Digital Industry Technologies',
            'Τουρκικών Σπουδών και Σύγχρονων Ασιατικών Σπουδών': 'Turkish Studies and Modern Asian Studies',
            'Φιλοσοφίας': 'Philosophy',
            'Φιλοσοφίας, Παιδαγωγικής και Ψυχολογίας': 'Philosophy, Pedagogy and Psychology',
            'Ψηφιακών Τεχνών και Κινηματογράφου': 'Digital Arts and Cinema',
            'Μηχανικών Τεχνολογίας Αεροσκαφών': 'Aircraft Technology Engineering',
            'Αγρονόμων και Τοπογράφων Μηχανικών - Μηχανικών Γεωπληροφορικής': 'Agricultural and Surveying Engineering - Geoinformatics Engineering',
            'Εφαρμοσμένων Μαθηματικών και Φυσικών Επιστημών': 'Applied Mathematics and Physical Sciences',
            'Μεταλλειολόγων - Μεταλλουργών Μηχανικών': 'Mining - Metallurgical Engineering',
            'Ναυπηγών Μηχανολόγων Μηχανικών': 'Naval Architects and Mechanical Engineering',
            'Διοίκησης Επιχειρήσεων και Τουρισμού': 'Business and Tourism Management',
            'Μουσικής Τεχνολογίας και Ακουστικής': 'Music Technology and Acoustics',
            'Μηχανικών Μουσικής Τεχνολογίας και Ακουστικής': 'Music Technology and Acoustics Engineering',
            'Μηχανικών Φυσικών Πόρων και Περιβάλλοντος': 'Natural Resources and Environment Engineering',
            'Πολιτικών Δομικών Έργων': 'Civil Construction Works',
            'Αρχειονομίας, Βιβλιοθηκονομίας και Μουσειολογίας': 'Archival Science, Library Science and Museology',
            'Ιστορίας': 'History',
            'Ξένων Γλωσσών, Μετάφρασης και Διερμηνείας': 'Foreign Languages, Translation and Interpretation',
            'Περιβάλλοντος': 'Environment',
            'Τεχνών Ήχου και Εικόνας': 'Sound and Image Arts',
            'Τουρισμού': 'Tourism',
            'Ψηφιακών Μέσων και Επικοινωνίας': 'Digital Media and Communication',
            'Συντήρησης Αρχαιοτήτων και Έργων Τέχνης': 'Conservation of Antiquities and Works of Art',
            'Τεχνολογίας Ήχου και Μουσικών Οργάνων': 'Sound Technology and Musical Instruments',
            'Τεχνολογιών Φυσικού Περιβάλλοντος': 'Natural Environment Technologies',
            'Διεθνείς και Ευρωπαϊκές Οικονομικές Σπουδές': 'International and European Economic Studies',
            'Μάρκετινγκ και Επικοινωνίας': 'Marketing and Communication',
            'Οικονομικής Επιστήμης': 'Economic Science',
            'Στατιστικής': 'Statistics',
            'Γεωγραφίας': 'Geography',
            'Επιστήμης Τροφίμων και Διατροφής': 'Food Science and Nutrition',
            'Επιστημών Προσχολικής Αγωγής και Εκπαιδευτικού Σχεδιασμού': 'Preschool Education Sciences and Educational Design',
            'Κοινωνικής Ανθρωπολογίας και Ιστορίας': 'Social Anthropology and History',
            'Μεσογειακών Σπουδών: Αρχαιολογία, Γλωσσολογία, Διεθνείς Σχέσεις': 'Mediterranean Studies: Archaeology, Linguistics, International Relations',
            'Μηχανικών Οικονομίας και Διοίκησης': 'Economics and Management Engineering',
            'Μηχανικών Πληροφοριακών και Επικοινωνιακών Συστημάτων': 'Information and Communication Systems Engineering',
            'Μηχανικών Σχεδίασης Προϊόντων και Συστημάτων': 'Product and Systems Design Engineering',
            'Ναυτιλίας και Επιχειρηματικών Υπηρεσιών': 'Shipping and Business Services',
            'Οικονομικής και Διοίκησης Τουρισμού': 'Tourism Economics and Management',
            'Πολιτισμικής Τεχνολογίας και Επικοινωνίας': 'Cultural Technology and Communication',
            'Στατιστικής και Αναλογιστικών - Χρηματοοικονομικών Μαθηματικών': 'Statistics and Actuarial - Financial Mathematics',
            'Ωκεανογραφίας και Θαλασσίων Βιοεπιστημών': 'Oceanography and Marine Biosciences',
            'Αρχειονομίας, Βιβλιοθηκονομίας και Συστημάτων Πληροφόρησης': 'Archival Science, Library Science and Information Systems',
            'Γραφιστικής και Οπτικής Επικοινωνίας': 'Graphic Design and Visual Communication',
            'Δημόσιας και Κοινοτικής Υγείας': 'Public and Community Health',
            'Διοίκησης Τουρισμού': 'Tourism Management',
            'Επιστημών Οίνου, Αμπέλου και Ποτών': 'Wine, Vine and Beverage Sciences',
            'Εργοθεραπείας': 'Occupational Therapy',
            'Ηλεκτρολόγων και Ηλεκτρονικών Μηχανικών': 'Electrical and Electronic Engineering',
            'Μηχανικών Βιοϊατρικής': 'Biomedical Engineering',
            'Μηχανικών Βιομηχανικής Σχεδίασης και Παραγωγής': 'Industrial Design and Production Engineering',
            'Μηχανικών Πληροφορικής και Υπολογιστών': 'Informatics and Computer Engineering',
            'Ναυπηγών Μηχανικών': 'Naval Architects',
            'Φωτογραφίας και Οπτικοακουστικών Τεχνών': 'Photography and Audiovisual Arts',
            'Ακτινολογίας και Ακτινοθεραπείας': 'Radiology and Radiotherapy',
            'Αυτοματισμού': 'Automation',
            'Γραφιστικής': 'Graphic Design',
            'Εμπορίας και Διαφήμισης': 'Commerce and Advertising',
            'Κλωστοϋφαντουργίας': 'Textile Engineering',
            'Μηχανικών Βιοϊατρικής Τεχνολογίας': 'Biomedical Technology Engineering',
            'Μηχανικών Ενεργειακής Τεχνολογίας': 'Energy Technology Engineering',
            'Μηχανικών Ηλεκτρονικών Υπολογιστικών Συστημάτων': 'Electronic Computing Systems Engineering',
            'Οδοντικής Τεχνολογίας': 'Dental Technology',
            'Οπτικής και Οπτομετρίας': 'Optics and Optometry',
            'Τεχνολογίας Γραφικών Τεχνών': 'Graphic Arts Technology',
            'Διεθνών και Ευρωπαϊκών Οικονομικών Σπουδών': 'International and European Economic Studies',
            'Επικοινωνίας και Ψηφιακών Μέσων': 'Communication and Digital Media',
            'Μηχανικών Ορυκτών Πόρων': 'Mineral Resources Engineering',
            'Παιδαγωγικό Νηπιαγωγών': 'Kindergarten Pedagogy',
            'Περιφερειακής και Διασυνοριακής Ανάπτυξης': 'Regional and Cross-Border Development',
            'Στατιστικής και Ασφαλιστικής Επιστήμης': 'Statistics and Insurance Science',
            'Βιομηχανικού Σχεδιασμού': 'Industrial Design',
            'Διεθνούς Εμπορίου': 'International Trade',
            'Μηχανικών Αντιρρύπανσης': 'Anti-Pollution Engineering',
            'Μηχανικών Γεωτεχνολογίας Περιβάλλοντος': 'Geoenvironmental Technology Engineering',
            'Βιοχημείας και Βιοτεχνολογίας': 'Biochemistry and Biotechnology',
            'Γεωπονίας - Αγροτεχνολογίας': 'Agriculture - Agricultural Technology',
            'Γεωπονίας, Ιχθυολογίας και Υδάτινου Περιβάλλοντος': 'Agriculture, Fisheries and Aquatic Environment',
            'Γεωπονίας, Φυτικής Παραγωγής και Αγροτικού Περιβάλλοντος': 'Agriculture, Plant Production and Agricultural Environment',
            'Γλωσσικών και Διαπολιτισμικών Σπουδών': 'Linguistic and Cross-Cultural Studies',
            'Δασολογίας, Επιστημών Ξύλου και Σχεδιασμού': 'Forestry, Wood Sciences and Design',
            'Δημόσιας και Ενιαίας Υγείας': 'Public and Unified Health',
            'Διαιτολογίας και Διατροφολογίας': 'Dietology and Nutrition Science',
            'Ιστορίας, Αρχαιολογίας και Κοινωνικής Ανθρωπολογίας': 'History, Archaeology and Social Anthropology',
            'Μηχανικών Χωροταξίας, Πολεοδομίας και Περιφερειακής Ανάπτυξης': 'Spatial Planning, Urban Planning and Regional Development Engineering',
            'Παιδαγωγικό Ειδικής Αγωγής': 'Special Education Pedagogy',
            'Παιδαγωγικό Προσχολικής Εκπαίδευσης': 'Preschool Education Pedagogy',
            'Πληροφορικής με Εφαρμογές στη Βιοϊατρική': 'Informatics with Applications in Biomedicine',
            'Πολιτισμού και Δημιουργικών Μέσων και Βιομηχανιών': 'Culture and Creative Media and Industries',
            'Συστημάτων Ενέργειας': 'Energy Systems',
            'Ψηφιακών Συστημάτων': 'Digital Systems',
            'Σχεδιασμού και Τεχνολογίας Ξύλου και Επίπλου': 'Wood and Furniture Design and Technology',
            'Βιολογικών Εφαρμογών και Τεχνολογιών': 'Biological Applications and Technologies',
            'Εικαστικών Τεχνών και Επιστημών Τέχνης': 'Fine Arts and Art Sciences',
            'Λογοθεραπείας': 'Speech Therapy',
            'Μηχανικών Επιστήμης Υλικών': 'Materials Science Engineering',
            'Μηχανικών Ηλεκτρονικών Υπολογιστών και Πληροφορικής': 'Electronic Computers and Informatics Engineering',
            'Λαϊκής και Παραδοσιακής Μουσικής': 'Folk and Traditional Music',
            'Επιστήμης και Μηχανικής Υλικών': 'Materials Science and Engineering',
            'Επιστήμης Υπολογιστών': 'Computer Science',
            'Εφαρμοσμένων Μαθηματικών': 'Applied Mathematics',
            'Φιλοσοφικών και Κοινωνικών Σπουδών': 'Philosophical and Social Studies',
            'Βαλκανικών, Σλαβικών και Ανατολικών Σπουδών': 'Balkan, Slavic and Eastern Studies',
            'Διεθνών και Ευρωπαϊκών Σπουδών': 'International and European Studies',
            'Εκπαιδευτικής και Κοινωνικής Πολιτικής': 'Educational and Social Policy',
            'Επιστήμη και Τεχνολογία Υπολογιστών': 'Computer Science and Technology',
            'Εφαρμοσμένης Πληροφορικής': 'Applied Informatics',
            'Μουσικής Επιστήμης και Τέχνης': 'Music Science and Art',
            'Πληροφοριακά Συστήματα': 'Information Systems',
            'Αλιείας και Υδατοκαλλιεργειών (πρώην Ζωικής Παραγωγής, Αλιείας και Υδατοκαλλιεργειών': 'Fisheries and Aquaculture (formerly Animal Production, Fisheries and Aquaculture)',
            'Διαχείρισης Περιβάλλοντος και Φυσικών Πόρων': 'Environment and Natural Resources Management',
            'Διαχείρισης Πολιτισμικού Περιβάλλοντος και Νέων Τεχνολογιών': 'Cultural Environment and New Technologies Management',
            'Διοίκησης Επιχειρήσεων Αγροτικών Προϊόντων και Τροφίμων': 'Agricultural Products and Food Business Management',
            'Επιστήμης Βιοσυστημάτων και Γεωργικής Μηχανικής': 'Biosystems Science and Agricultural Engineering',
            'Επιστήμης των Υλικών': 'Materials Science',
            'Επιστημών της Εκπαίδευσης και Κοινωνικής Εργασίας': 'Education Sciences and Social Work',
            'Επιστημών της Εκπαίδευσης και της Αγωγής στην Προσχολική Ηλικία': 'Education Sciences and Preschool Training',
            'Ηλεκτρολόγων Μηχανικών και Τεχνολογίας Υπολογιστών': 'Electrical Engineering and Computer Technology',
            'Ιστορίας - Αρχαιολογίας': 'History - Archaeology',
            'Μηχανολόγων και Αεροναυπηγών Μηχανικών': 'Mechanical and Aeronautical Engineering',
            'Μουσειολογίας': 'Museology',
            'Τεχνολογίας Αλιείας και Υδατοκαλλιεργειών': 'Fisheries and Aquaculture Technology',
            'Τουριστικών Επιχειρήσεων': 'Tourism Businesses',
            'Βιομηχανικής Διοίκησης και Τεχνολογίας': 'Industrial Management and Technology',
            'Ναυτιλιακών Σπουδών': 'Shipping Studies',
            'Τουριστικών Σπουδών': 'Tourism Studies',
            'Χρηματοοικονομικής και Τραπεζικής Διοικητικής': 'Finance and Banking Administration',
            'Επιστήμης Διατροφής και Διαιτολογίας': 'Nutrition and Dietetics Science',
            'Ιστορίας, Αρχαιολογίας και Διαχείρισης Πολιτισμικών Αγαθών': 'History, Archaeology and Cultural Goods Management',
            'Κοινωνικής και Εκπαιδευτικής Πολιτικής': 'Social and Educational Policy',
            'Οργάνωσης και Διαχείρισης Αθλητισμού': 'Sports Organization and Management',
            'Παραστατικών και Ψηφιακών Τεχνών': 'Performing and Digital Arts',
            'Πολιτικής Επιστήμης και Διεθνών Σχέσεων': 'Political Science and International Relations',
            'Διοίκησης Μονάδων Υγείας και Πρόνοιας': 'Health Units and Welfare Management',
            'Τοπικής Αυτοδιοίκησης': 'Local Government',
            'Δημόσιας Διοίκησης': 'Public Administration',
            'Επικοινωνίας, Μέσων και Πολιτισμού': 'Communication, Media and Culture',
            'Κοινωνικής Ανθρωπολογίας': 'Social Anthropology',
            'Οικονομικής και Περιφερειακής Ανάπτυξης': 'Economic and Regional Development',
            'Πολιτικής Επιστήμης και Ιστορίας': 'Political Science and History',
            'Χημικών Μηχανικών και Μηχανικών Περιβάλλοντος': 'Chemical Engineering and Environmental Engineering',
            'Στρατιωτικών Επιστημών': 'Military Sciences',
            'Αεροπορικών Επιστημών': 'Aviation Sciences',
            'Ναυτικών Επιστημών': 'Naval Sciences',
            'Επιστήμης Διαιτολογίας - Διατροφής': 'Dietology - Nutrition Science',
            'Οικονομίας και Βιώσιμης Ανάπτυξης (πρώην Οικιακής Οικονομίας και Οικολογίας': 'Economy and Sustainable Development (formerly Home Economics and Ecology)',
            'Πληροφορικής και Τηλεματικής': 'Informatics and Telematics',
            'Αειφορικής Γεωργίας': 'Sustainable Agriculture',
            'Γεωπονίας  - νέο': 'Agriculture - new',
            'Δημόσια Διοίκηση':	'Public Administration',
            'Διοίκηση Τουρισμού': 'Tourism Management',
            'Ισπανικής Γλώσσας και Πολιτισμού': 'Spanish Language and Culture',
            'Σπουδές στις Φυσικές Επιστήμες': 'Studies in Natural Sciences',
            'Σπουδές στον Ελληνικό Πολιτισμό': 'Studies in Hellenic (Greek) Culture',
            'Σπουδές στον Ευρωπαϊκό Πολιτισμό': 'Studies in European Culture',
            'Σπουδές Κινηματογραφικής Γραφής, Πρακτικής και Έρευνας': 'Studies in Film Writing, Practice, and Research',
            'Αγγλόγλωσσο Προπτυχιακό Πρόγραμμα Σπουδών του Τμήματος Ιατρικής ΑΠΘ': 'Medical English Program in AUTH',
            'Διεθνές Προπτυχιακό Πρόγραμμα Ιατρικής Σχολής': 'Interational Program in Medicine in University of Crete'
        }
        
    def get_map(self) -> dict:
        return self.__programme_map
    
    def map(self, program: str) -> str:
        if program.isascii():
            return program
        if  self.__programme_map.get(program, "Unknown") == "Unknown":
            print(f"Warning: program '{program}' not found in mapping.")
        return self.__programme_map.get(program, "Unknown")
    

class InstituteMapper:

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

    
    def get_map(self):
        """
        Returns the initialized dictionary mapping institution IDs to English names.
        """
        return self._institute_map
    
    
    def map(self, institute_id: str) -> str:
        """
        Maps a given institution ID to its English name.
        Args:
            institute_id (str): The institution ID to map.
        Returns:
            str: The English name of the institution or "Unknown Institute" if not found.
        """
        return self._institute_map.get(institute_id, "Unknown Institute")
    
    def get_institutes_ids(self):
        """
        Returns a list of all institution IDs in the mapping.
        """
        return list(self._institute_map.keys())


class AcademicYearMapper:

    def __init__(self):
        """
        Initializes the year_map dictionary as an instance attribute.
        """
        self._year_map = {
            '2022': '2020-2021',
            '2023': '2021-2022',
            '2024': '2022-2023',
            '2025': '2023-2024',
        }

    
    def get_map(self):
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


url = "https://www.ethaae.gr/el/dedomena-aei/foitites-aei"

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Origin": "https://www.ethaae.gr",
    "Referer": "https://www.ethaae.gr/el/dedomena-aei/foitites-aei",
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


def save_to_csv(data: List[List[Dict]], filename: str):
    """Save the list of dictionaries to a CSV file.
    
    Args:
        data (list): List of dictionaries containing the scraped data.
        filename (str): The name of the CSV file to save the data.
    """
    flat_data = [item for sublist in data for item in sublist]
    df = pd.DataFrame(flat_data)
    df.to_csv(filename, index=False)


def scrape_data(soup: BeautifulSoup):
    """Scrape data from the BeautifulSoup object and return a list of dictionaries.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the HTML content.
    """
    soup_data = soup.find_all('div', class_='stats-item')   # get all the programmes
    scraped_data = []
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

collected_data = []
institude_mappper = InstituteMapper()
for year in range(2022, 2026):    
    for inst_id in InstituteMapper().get_institutes_ids():
        data["filter[collectionyear]"] = str(year)
        data["filter[instituteid]"] = str(inst_id)

        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        collected_data.append(scrape_data(soup))

save_to_csv(collected_data, f"hahe_all_21_24.csv")
