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

    def analyze_and_summarize(self, question: str, results: List[Dict], drug_name: str, max_tokens: int = 500) -> str:
        """
        Arama sonuçlarını daha akıllı analiz eder ve özetler.
        """
        if not results:
            return "Üzgünüm, arama sonuçları bulunamadı. Başka bir şekilde size nasıl yardımcı olabilirim?"

        # Sonuçları birleştir ve önemli bilgileri vurgula
        context = ""
        for result in results[:5]:
            # Başlık ve snippet'i temizle
            title = re.sub(r'\s+', ' ', result['title']).strip()
            snippet = re.sub(r'\s+', ' ', result['snippet']).strip()
            
            # Güvenilirlik göstergesi ekle
            reliability = "Güvenilir Kaynak: " if result["domain"] in self.trusted_domains else ""
            
            context += f"\nKaynak: {reliability}{result['domain']}\n"
            context += f"Başlık: {title}\n"
            context += f"Özet: {snippet}\n"
            context += f"URL: {result['url']}\n"
            context += "-" * 50 + "\n"

        prompt = f"""
Aşağıdaki güvenilir kaynaklardan alınan bilgileri kullanarak, {drug_name} ilacı hakkında 
kapsamlı ve anlaşılır bir yanıt oluştur.

Soru: {question}

Kaynaklar:
{context}

Yanıtı oluştururken:
1. Sadece güvenilir kaynaklardan gelen bilgileri kullan
2. Bilgileri önem sırasına göre düzenle
3. Teknik terimleri basit bir dille açıkla
4. Önemli uyarıları vurgula
5. Her zaman bir doktora danışılması gerektiğini belirt

Yanıt:
"""

        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Sen güvenilir bir sağlık bilgi asistanısın. "
                 "Sadece kanıtlanmış ve güvenilir kaynaklardan gelen bilgileri kullanırsın. "
                 "Bilgileri açık ve anlaşılır bir dille aktarırsın."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5,
        )

        return response.choices[0].message.content.strip()

    def _simplify_urls(self, text: str) -> str:
        def simplify(match):
            url = match.group(0)
            parsed_url = urlparse(url)
            return parsed_url.netloc

        return re.sub(r'https?://[^\s]+', simplify, text)