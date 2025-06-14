import requests
import json
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict
import logging


class Drug:
    """
    FDA'dan ilaç bilgilerini çeken ve bu bilgileri temizleyen sınıf.
    """

    def __init__(self, drug_name: str):
        """
        Drug sınıfının yapıcısı.

        Args:
            drug_name (str): İngilizce etken madde/ilaç adı.
        """
        self.drug_name: str = drug_name  # İngilizce etken madde/ilaç adı
        self.brand_name: Optional[str] = None
        self.generic_name: Optional[str] = None
        self.substance_name: Optional[str] = None
        self.indications_and_usage: Optional[str] = None
        self.warnings: Optional[str] = None
        self.dosage_and_administration: Optional[str] = None
        self.adverse_reactions: Optional[str] = None

    def _fetch_fda_data(self) -> Optional[Dict]:
        """
        FDA openFDA API'sinden ilaç bilgilerini çeker.

        Returns:
            Optional[Dict]: API'den dönen JSON verisi veya None (hata durumunda).
        """
        url: str = "https://api.fda.gov/drug/label.json"
        params: Dict[str, str] = {
            "search": f'openfda.substance_name:"{self.drug_name}"',
            "limit": 1
        }
        try:
            response = requests.get(url, params=params)
            logging.info(f"API URL: {response.url}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"FDA API'sine erişimde hata: {e}")
            logging.error(f"Hata Detayı: {e.response.text if e.response else None}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"FDA API yanıtını işlerken hata: {e}")
            return None

    def get_fda_info(self) -> bool:
        """
        FDA verilerini çeker ve sınıf özelliklerine kaydeder.

        Returns:
            bool: Başarılı ise True, başarısız ise False.
        """
        fda_data: Optional[Dict] = self._fetch_fda_data()
        if not fda_data or 'results' not in fda_data or not fda_data['results']:
            print(f"FDA'da '{self.drug_name}' için bilgi bulunamadı.")
            return False

        try:
            result: Dict = fda_data['results'][0]
            openfda: Dict = result.get('openfda', {})

            # Bilgileri sınıfın özniteliklerine kaydet
            self.brand_name = openfda.get('brand_name', [None])[0] if openfda.get('brand_name') else None
            self.generic_name = openfda.get('generic_name', [None])[0] if openfda.get('generic_name') else None
            self.substance_name = openfda.get('substance_name', [None])[0] if openfda.get('substance_name') else None
            # Aşağıdaki satırları GÜNCELLEDİK
            self.indications_and_usage = result.get('indications_and_usage', [None])[0] if result.get('indications_and_usage') else None
            self.warnings = result.get('warnings', [None])[0] if result.get('warnings') else None
            self.dosage_and_administration = result.get('dosage_and_administration', [None])[0] if result.get('dosage_and_administration') else None
            self.adverse_reactions = result.get('adverse_reactions', [None])[0] if result.get('adverse_reactions') else None
            return True

        except (KeyError, IndexError, TypeError) as e:
            print(f"FDA verisini işlerken hata: {e}")
            return False

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """
        Metni temizleyen yardımcı fonksiyon.

        Args:
            text (Optional[str]): Temizlenecek metin.

        Returns:
            Optional[str]: Temizlenmiş metin veya None.
        """
        if text:
            soup: BeautifulSoup = BeautifulSoup(text, 'html.parser')
            text: str = soup.get_text(separator=" ")
            text: str = re.sub(r'\s+', ' ', text).strip()
            return text
        return None

    def clean_data(self):
        """Çekilen ilaç bilgilerini temizler (HTML etiketleri, vb.)."""

        self.indications_and_usage = self._clean_text(self.indications_and_usage)
        self.warnings = self._clean_text(self.warnings)
        self.dosage_and_administration = self._clean_text(self.dosage_and_administration)
        self.adverse_reactions = self._clean_text(self.adverse_reactions)
        # ... Diğer metin türündeki öznitelikleri de temizle ...


if __name__ == '__main__':
    # Örnek kullanım
    drug = Drug("Enalapril")  # Örnek ilaç adı
    if drug.get_fda_info():
        print(f"İlaç Adı: {drug.substance_name}")
        print(f"Endikasyonları: {drug.indications_and_usage}")
        drug.clean_data()
        print(f"Temizlenmiş Endikasyonları: {drug.indications_and_usage}")
    else:
        print("İlaç bilgisi alınamadı.")