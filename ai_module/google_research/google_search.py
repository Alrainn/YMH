import requests
from typing import List, Dict, Optional
import os
from openai import OpenAI
from urllib.parse import urlparse
import re
import logging

openai_api_key = os.environ.get("OPENAI_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")
cse_id = os.environ.get("GOOGLE_CSE_ID")


class GoogleSearch:
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.drug_corrections = {
            "ariprazol": "aripiprazol",
            "ariprazole": "aripiprazole",
            # Diğer düzeltmeler eklenebilir
        }
        self.emotion_patterns = {
            "endişe": ["endişe", "korku", "tedirgin", "risk", "tehlike", "yan etki"],
            "acı": ["ağrı", "sızı", "acı", "rahatsızlık"],
            "umut": ["iyileşme", "tedavi", "düzelme", "fayda"],
            "belirsizlik": ["emin değilim", "bilmiyorum", "acaba", "mı", "mi"]
        }
        # Güvenilir sağlık siteleri
        self.trusted_domains = [
            "saglik.gov.tr",
            "titck.gov.tr",
            "ilacrehberi.com",
            "medikalakademi.com.tr",
            "ilacprospektusu.com",
            "medikal.com.tr",
            "acibadem.com.tr",
            "memorial.com.tr",
            "amerikanhastanesi.org"
        ]
        
        # İlaç bilgisi için önemli anahtar kelimeler
        self.drug_info_keywords = {
            "genel": ["prospektüs", "kullanma talimatı", "ilaç bilgisi", "etken madde"],
            "yan_etki": ["yan etki", "istenmeyen etki", "advers etki", "komplikasyon"],
            "kullanım": ["kullanım", "doz", "dozaj", "nasıl kullanılır", "ne kadar"],
            "uyarı": ["uyarı", "dikkat", "önlem", "kontrendikasyon", "etkileşim"]
        }

    def detect_emotion(self, text: str) -> List[str]:
        """Metindeki duygu durumlarını tespit eder."""
        detected_emotions = []
        lower_text = text.lower()
        
        for emotion, patterns in self.emotion_patterns.items():
            if any(pattern in lower_text for pattern in patterns):
                detected_emotions.append(emotion)
        
        return detected_emotions

    def create_empathetic_response(self, response: str, emotions: List[str]) -> str:
        """Yanıtı duygusal bağlama göre şekillendirir."""
        empathy_prefixes = {
            "endişe": "Endişelerinizi anlıyorum. Size güvenilir bilgiler sunmak istiyorum. ",
            "acı": "Yaşadığınız rahatsızlığı anlıyorum. Size yardımcı olmak için bulduğum bilgiler şöyle: ",
            "umut": "Olumlu gelişmeler görmek güzel. Araştırmalarıma göre: ",
            "belirsizlik": "Size bu konuda yardımcı olmak isterim. İşte bulduğum bilgiler: "
        }
        
        if not emotions:
            return response
            
        prefix = empathy_prefixes.get(emotions[0], "")
        return prefix + response

    def create_search_queries(self, question: str, drug_name: str, num_queries: int = 3) -> List[str]:
        """
        Daha akıllı ve odaklı arama sorguları oluşturur.
        """
        # İlaç adı düzeltmesi
        corrected_drug_name = self.drug_corrections.get(drug_name.lower(), drug_name)

        # Soru türünü belirle
        question_lower = question.lower()
        query_type = None
        
        for key, keywords in self.drug_info_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                query_type = key
                break
        
        # Sorgu şablonları
        query_templates = {
            "genel": [
                f"{corrected_drug_name} prospektüs",
                f"{corrected_drug_name} ilaç bilgisi kullanım",
                f"{corrected_drug_name} etken madde endikasyon"
            ],
            "yan_etki": [
                f"{corrected_drug_name} yan etkileri",
                f"{corrected_drug_name} istenmeyen etkiler güvenlik",
                f"{corrected_drug_name} yan etki risk"
            ],
            "kullanım": [
                f"{corrected_drug_name} nasıl kullanılır doz",
                f"{corrected_drug_name} kullanım talimatı dozaj",
                f"{corrected_drug_name} kullanım şekli süre"
            ],
            "uyarı": [
                f"{corrected_drug_name} kullanım uyarıları",
                f"{corrected_drug_name} dikkat edilmesi gerekenler",
                f"{corrected_drug_name} kontrendikasyon etkileşim"
            ]
        }
        
        # Sorgu türüne göre şablonları seç
        if query_type and query_type in query_templates:
            queries = query_templates[query_type]
        else:
            queries = query_templates["genel"]
        
        return queries[:num_queries]

    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Geliştirilmiş web arama fonksiyonu.
        """
        url = "https://www.googleapis.com/customsearch/v1"

        # Sorguyu temizle ve optimize et
        clean_query = query.replace('"', '').strip()
        
        # Site kısıtlaması ekle
        site_restriction = " OR ".join(f"site:{domain}" for domain in self.trusted_domains)
        final_query = f"({clean_query}) ({site_restriction})"

        params = {
            "q": final_query,
            "cx": self.cse_id,
            "key": self.api_key,
            "num": num_results * 2,  # Daha fazla sonuç al, sonra filtrele
            "lr": "lang_tr",
            "gl": "tr",
            "fields": "items(title,link,snippet)"
        }

        try:
            print(f"Arama sorgusu: {final_query}")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                print(f"API Hatası: {data['error']['message']}")
                return []

            if "items" not in data:
                print(f"Sonuç bulunamadı. Sorgu: '{final_query}'")
                return []

            # Sonuçları filtrele ve sırala
            results = []
            for item in data.get("items", []):
                domain = urlparse(item["link"]).netloc
                if any(trusted in domain for trusted in self.trusted_domains):
                    results.append({
                        "title": item["title"],
                        "url": item["link"],
                        "snippet": item.get("snippet", ""),
                        "domain": domain
                    })
            
            # En güvenilir sonuçları önce göster
            results.sort(key=lambda x: self.trusted_domains.index(x["domain"]) 
                        if x["domain"] in self.trusted_domains else len(self.trusted_domains))
            
            return results[:num_results]

        except requests.exceptions.RequestException as e:
            print(f"Google API isteği hatası: {e}")
            return []
        except ValueError as e:
            print(f"JSON işleme hatası: {e}")
            return []

    def analyze_and_summarize(self, query: str, results: List[Dict], drug_name: str) -> str:
        """
        Arama sonuçlarını analiz eder ve yapılandırılmış bir özet oluşturur.
        """
        try:
            # İlaç adını doğrula
            confirmed_drug_name = self._confirm_drug_name(drug_name, results)
            if not confirmed_drug_name:
                return f"Üzgünüm, {drug_name} hakkında güvenilir bilgi bulamadım. Lütfen ilaç adının doğru yazıldığından emin olun."
            
            # Arama sonuçlarını birleştir
            combined_text = " ".join([result.get('snippet', '') for result in results])
            
            # İlaç bilgilerini çıkar
            drug_info = self._extract_drug_info(combined_text, confirmed_drug_name)
            
            # Yanıtı yapılandır
            response = f"""
{confirmed_drug_name} Hakkında Detaylı Bilgi:

1. İlacın Tam Adı ve Etken Maddesi:
   - İlaç Adı: {drug_info['full_name']}
   - Etken Madde: {drug_info['active_ingredient']}
   - Farmakolojik Sınıf: {drug_info['pharmacological_class']}

2. Kullanım Amacı ve Endikasyonlar:
   {self._format_list(drug_info['indications'])}

3. Yan Etkiler:
   a) Çok Yaygın Yan Etkiler (10 hastadan 1'inden fazlasında):
   {self._format_side_effects(drug_info['side_effects']['very_common'])}
   
   b) Yaygın Yan Etkiler (100 hastadan 1'inden fazlasında):
   {self._format_side_effects(drug_info['side_effects']['common'])}
   
   c) Nadir Yan Etkiler (1000 hastadan 1'inden az):
   {self._format_side_effects(drug_info['side_effects']['rare'])}
   
   d) Çok Nadir Yan Etkiler (10.000 hastadan 1'inden az):
   {self._format_side_effects(drug_info['side_effects']['very_rare'])}

4. Önemli Uyarılar ve Kontrendikasyonlar:
   {self._format_list(drug_info['warnings'])}

5. İlaç Etkileşimleri:
   {self._format_list(drug_info['interactions'])}

6. Dozaj Bilgileri:
   {self._format_list(drug_info['dosage'])}

7. Kullanım Süresi:
   {self._format_list(drug_info['duration'])}

8. Özel Durumlar:
   a) Hamilelik:
   {self._format_list(drug_info['special_cases']['pregnancy'])}
   
   b) Emzirme:
   {self._format_list(drug_info['special_cases']['breastfeeding'])}
   
   c) Yaşlılar:
   {self._format_list(drug_info['special_cases']['elderly'])}
   
   d) Çocuklar:
   {self._format_list(drug_info['special_cases']['children'])}

9. Doktora Ne Zaman Başvurulmalı:
   {self._format_list(drug_info['when_to_consult_doctor'])}
"""
            return response.strip()
        except Exception as e:
            print(f"Özet oluşturma sırasında hata: {str(e)}")
            return "Üzgünüm, bilgileri işlerken bir hata oluştu."

    def _confirm_drug_name(self, drug_name: str, results: List[Dict]) -> Optional[str]:
        """
        İlaç adını doğrular ve tam adını döndürür.
        """
        try:
            # Arama sonuçlarında ilaç adını ara
            for result in results:
                snippet = result.get('snippet', '').lower()
                if drug_name.lower() in snippet:
                    # İlaç adını ve etken maddesini çıkar
                    lines = snippet.split('\n')
                    for line in lines:
                        if drug_name.lower() in line.lower():
                            # Parantez içindeki etken maddeyi bul
                            match = re.search(r'\((.*?)\)', line)
                            if match:
                                return f"{drug_name} ({match.group(1)})"
                            return drug_name
            return None
        except Exception as e:
            print(f"İlaç adı doğrulama sırasında hata: {str(e)}")
            return None

    def _extract_drug_info(self, text: str, drug_name: str) -> Dict:
        """
        Metinden ilaç bilgilerini çıkarır.
        """
        try:
            # İlaç bilgilerini çıkar
            info = {
                'full_name': drug_name,
                'active_ingredient': self._extract_active_ingredient(text),
                'pharmacological_class': self._extract_pharmacological_class(text),
                'indications': self._extract_indications(text),
                'side_effects': self._extract_side_effects(text),
                'warnings': self._extract_warnings(text),
                'interactions': self._extract_interactions(text),
                'dosage': self._extract_dosage(text),
                'duration': self._extract_duration(text),
                'special_cases': {
                    'pregnancy': self._extract_pregnancy_info(text),
                    'breastfeeding': self._extract_breastfeeding_info(text),
                    'elderly': self._extract_elderly_info(text),
                    'children': self._extract_children_info(text)
                },
                'when_to_consult_doctor': self._extract_when_to_consult_doctor(text)
            }
            return info
        except Exception as e:
            print(f"İlaç bilgisi çıkarma sırasında hata: {str(e)}")
            return {}

    def _format_list(self, items: List[str]) -> str:
        """
        Liste öğelerini formatlar.
        """
        if not items:
            return "   - Bilgi bulunamadı."
        return "\n".join([f"   - {item}" for item in items])

    def _format_side_effects(self, side_effects: List[Dict]) -> str:
        """
        Yan etkileri formatlar.
        """
        if not side_effects:
            return "   - Bilgi bulunamadı."
        
        formatted = []
        for effect in side_effects:
            formatted.append(f"   - {effect['name']}")
            formatted.append(f"     * Açıklama: {effect['description']}")
            formatted.append(f"     * Ortaya Çıkış: {effect['onset']}")
            formatted.append(f"     * Süre: {effect['duration']}")
        
        return "\n".join(formatted)

    def _simplify_urls(self, text: str) -> str:
        def simplify(match):
            url = match.group(0)
            parsed_url = urlparse(url)
            return parsed_url.netloc

        return re.sub(r'https?://[^\s]+', simplify, text)

    def _extract_active_ingredient(self, text: str) -> str:
        """
        Metinden etken maddeyi çıkarır.
        """
        try:
            # Etken madde için yaygın kalıpları ara
            patterns = [
                r'etken madde[:\s]+([^.\n]+)',
                r'aktif madde[:\s]+([^.\n]+)',
                r'\(([^)]+)\)\s*etken madde',
                r'\(([^)]+)\)\s*aktif madde'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            return "Bilgi bulunamadı"
        except Exception as e:
            print(f"Etken madde çıkarma hatası: {str(e)}")
            return "Bilgi bulunamadı"

    def _extract_pharmacological_class(self, text: str) -> str:
        """
        Metinden farmakolojik sınıfı çıkarır.
        """
        try:
            patterns = [
                r'farmakolojik sınıf[:\s]+([^.\n]+)',
                r'ilaç grubu[:\s]+([^.\n]+)',
                r'grup[:\s]+([^.\n]+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            return "Bilgi bulunamadı"
        except Exception as e:
            print(f"Farmakolojik sınıf çıkarma hatası: {str(e)}")
            return "Bilgi bulunamadı"

    def _extract_indications(self, text: str) -> List[str]:
        """
        Metinden endikasyonları çıkarır.
        """
        try:
            patterns = [
                r'endikasyonlar[:\s]+([^.\n]+)',
                r'kullanım amacı[:\s]+([^.\n]+)',
                r'kullanıldığı durumlar[:\s]+([^.\n]+)'
            ]
            
            indications = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    indication = match.group(1).strip()
                    if indication:
                        indications.append(indication)
            
            return indications if indications else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Endikasyon çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_side_effects(self, text: str) -> Dict[str, List[Dict]]:
        """
        Metinden yan etkileri çıkarır ve kategorize eder.
        """
        try:
            side_effects = {
                'very_common': [],
                'common': [],
                'rare': [],
                'very_rare': []
            }
            
            # Yan etki bölümlerini bul
            sections = {
                'very_common': r'çok yaygın[^:]*:([^.]*)',
                'common': r'yaygın[^:]*:([^.]*)',
                'rare': r'nadir[^:]*:([^.]*)',
                'very_rare': r'çok nadir[^:]*:([^.]*)'
            }
            
            for category, pattern in sections.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    effects_text = match.group(1).strip()
                    effects = [e.strip() for e in effects_text.split(',') if e.strip()]
                    
                    for effect in effects:
                        side_effects[category].append({
                            'name': effect,
                            'description': self._get_side_effect_description(effect, text),
                            'onset': self._get_side_effect_onset(effect, text),
                            'duration': self._get_side_effect_duration(effect, text)
                        })
            
            return side_effects
        except Exception as e:
            print(f"Yan etki çıkarma hatası: {str(e)}")
            return {
                'very_common': [],
                'common': [],
                'rare': [],
                'very_rare': []
            }

    def _get_side_effect_description(self, effect: str, text: str) -> str:
        """
        Yan etkinin açıklamasını bulur.
        """
        try:
            pattern = f"{effect}[^.]*\.([^.]*)"
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else "Açıklama bulunamadı"
        except Exception:
            return "Açıklama bulunamadı"

    def _get_side_effect_onset(self, effect: str, text: str) -> str:
        """
        Yan etkinin ne zaman ortaya çıktığını bulur.
        """
        try:
            pattern = f"{effect}[^.]*ortaya çık[^.]*\.([^.]*)"
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else "Bilgi bulunamadı"
        except Exception:
            return "Bilgi bulunamadı"

    def _get_side_effect_duration(self, effect: str, text: str) -> str:
        """
        Yan etkinin ne kadar sürdüğünü bulur.
        """
        try:
            pattern = f"{effect}[^.]*sür[^.]*\.([^.]*)"
            match = re.search(pattern, text, re.IGNORECASE)
            return match.group(1).strip() if match else "Bilgi bulunamadı"
        except Exception:
            return "Bilgi bulunamadı"

    def _extract_warnings(self, text: str) -> List[str]:
        """
        Metinden uyarıları çıkarır.
        """
        try:
            patterns = [
                r'uyarılar[:\s]+([^.\n]+)',
                r'dikkat edilmesi gerekenler[:\s]+([^.\n]+)',
                r'önlemler[:\s]+([^.\n]+)'
            ]
            
            warnings = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    warning = match.group(1).strip()
                    if warning:
                        warnings.append(warning)
            
            return warnings if warnings else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Uyarı çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_interactions(self, text: str) -> List[str]:
        """
        Metinden ilaç etkileşimlerini çıkarır.
        """
        try:
            patterns = [
                r'etkileşim[:\s]+([^.\n]+)',
                r'ilaç etkileşimi[:\s]+([^.\n]+)',
                r'diğer ilaçlarla[:\s]+([^.\n]+)'
            ]
            
            interactions = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    interaction = match.group(1).strip()
                    if interaction:
                        interactions.append(interaction)
            
            return interactions if interactions else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Etkileşim çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_dosage(self, text: str) -> List[str]:
        """
        Metinden dozaj bilgilerini çıkarır.
        """
        try:
            patterns = [
                r'doz[:\s]+([^.\n]+)',
                r'dozaj[:\s]+([^.\n]+)',
                r'kullanım miktarı[:\s]+([^.\n]+)'
            ]
            
            dosages = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    dosage = match.group(1).strip()
                    if dosage:
                        dosages.append(dosage)
            
            return dosages if dosages else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Dozaj çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_duration(self, text: str) -> List[str]:
        """
        Metinden kullanım süresini çıkarır.
        """
        try:
            patterns = [
                r'süre[:\s]+([^.\n]+)',
                r'kullanım süresi[:\s]+([^.\n]+)',
                r'ne kadar süre[:\s]+([^.\n]+)'
            ]
            
            durations = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    duration = match.group(1).strip()
                    if duration:
                        durations.append(duration)
            
            return durations if durations else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Süre çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_pregnancy_info(self, text: str) -> List[str]:
        """
        Metinden hamilelik bilgilerini çıkarır.
        """
        try:
            patterns = [
                r'hamilelik[:\s]+([^.\n]+)',
                r'gebelik[:\s]+([^.\n]+)',
                r'hamilelerde[:\s]+([^.\n]+)'
            ]
            
            info = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    item = match.group(1).strip()
                    if item:
                        info.append(item)
            
            return info if info else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Hamilelik bilgisi çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_breastfeeding_info(self, text: str) -> List[str]:
        """
        Metinden emzirme bilgilerini çıkarır.
        """
        try:
            patterns = [
                r'emzirme[:\s]+([^.\n]+)',
                r'laktasyon[:\s]+([^.\n]+)',
                r'emzirenlerde[:\s]+([^.\n]+)'
            ]
            
            info = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    item = match.group(1).strip()
                    if item:
                        info.append(item)
            
            return info if info else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Emzirme bilgisi çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_elderly_info(self, text: str) -> List[str]:
        """
        Metinden yaşlılar için bilgileri çıkarır.
        """
        try:
            patterns = [
                r'yaşlılarda[:\s]+([^.\n]+)',
                r'ileri yaş[:\s]+([^.\n]+)',
                r'yaşlı hastalarda[:\s]+([^.\n]+)'
            ]
            
            info = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    item = match.group(1).strip()
                    if item:
                        info.append(item)
            
            return info if info else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Yaşlı bilgisi çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_children_info(self, text: str) -> List[str]:
        """
        Metinden çocuklar için bilgileri çıkarır.
        """
        try:
            patterns = [
                r'çocuklarda[:\s]+([^.\n]+)',
                r'pediatrik[:\s]+([^.\n]+)',
                r'çocuk hastalarda[:\s]+([^.\n]+)'
            ]
            
            info = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    item = match.group(1).strip()
                    if item:
                        info.append(item)
            
            return info if info else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Çocuk bilgisi çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]

    def _extract_when_to_consult_doctor(self, text: str) -> List[str]:
        """
        Metinden doktora ne zaman başvurulması gerektiğini çıkarır.
        """
        try:
            patterns = [
                r'doktora başvurun[:\s]+([^.\n]+)',
                r'doktora danışın[:\s]+([^.\n]+)',
                r'doktora bildirin[:\s]+([^.\n]+)'
            ]
            
            info = []
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    item = match.group(1).strip()
                    if item:
                        info.append(item)
            
            return info if info else ["Bilgi bulunamadı"]
        except Exception as e:
            print(f"Doktor başvurusu bilgisi çıkarma hatası: {str(e)}")
            return ["Bilgi bulunamadı"]