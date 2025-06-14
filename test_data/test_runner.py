import json
import os
import sys
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Ana dizini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai import AIAssistant

# .env dosyasını yükle
load_dotenv()

class TestRunner:
    def __init__(self, api_key: str):
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("Lütfen geçerli bir OpenAI API anahtarı girin")
        self.ai = AIAssistant(api_key=api_key)
        self.test_results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "details": [],
            "timestamp": datetime.now().isoformat(),
            "test_suite": "AI Assistant Test Suite"
        }
        self.report_dir = "test_reports"
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    def load_test_cases(self, file_path: str) -> List[Dict[str, Any]]:
        """Test senaryolarını JSON dosyasından yükler."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('test_cases', [])

    def check_content_match(self, response: str, expected_phrases: List[str]) -> bool:
        """Yanıt içeriğinin beklenen ifadeleri içerip içermediğini kontrol eder."""
        if not response:
            return False
        response_lower = response.lower()
        return all(phrase.lower() in response_lower for phrase in expected_phrases)

    def save_test_results(self):
        """Test sonuçlarını JSON dosyasına kaydeder."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        filepath = os.path.join(self.report_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        return filepath

    def generate_html_report(self):
        """Test sonuçlarından HTML raporu oluşturur."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{timestamp}.html"
        filepath = os.path.join(self.report_dir, filename)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>AI Assistant Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .test-case {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .passed {{ background-color: #e6ffe6; }}
                .failed {{ background-color: #ffe6e6; }}
                .error {{ background-color: #fff0e6; }}
                .details {{ margin-left: 20px; }}
            </style>
        </head>
        <body>
            <h1>AI Assistant Test Report</h1>
            <div class="summary">
                <h2>Test Summary</h2>
                <p>Total Tests: {self.test_results['total']}</p>
                <p>Passed: {self.test_results['passed']}</p>
                <p>Failed: {self.test_results['failed']}</p>
                <p>Timestamp: {self.test_results['timestamp']}</p>
            </div>
            <h2>Test Details</h2>
        """
        
        for detail in self.test_results['details']:
            status_class = detail['status'].lower()
            html_content += f"""
            <div class="test-case {status_class}">
                <h3>Test ID: {detail['id']}</h3>
                <p>Status: {detail['status']}</p>
                <p>Description: {detail['description']}</p>
                <div class="details">
            """
            
            if detail['status'] == "ERROR":
                html_content += f"<p>Error: {detail['error']}</p>"
            else:
                for key, value in detail.items():
                    if key not in ['id', 'status', 'description']:
                        html_content += f"<p>{key}: {value}</p>"
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath

    def run_drug_tests(self):
        """İlaç sorguları testlerini çalıştırır."""
        test_cases = self.load_test_cases('test_data/drug_queries.json')
        for test_case in test_cases:
            self.test_results['total'] += 1
            try:
                response = self.ai.get_response(test_case['input'])
                if response is None:
                    raise Exception("API yanıt vermedi")
                
                # Duygu ve yoğunluk kontrolü
                emotions, intensity = self.ai.detect_emotion_and_intensity(test_case['input'])
                if emotions is None or intensity is None:
                    raise Exception("Duygu analizi başarısız")
                
                emotion_match = all(e in emotions for e in test_case['expected_emotions'])
                intensity_match = abs(intensity - test_case['expected_intensity']) <= 1
                
                # İlaç adı kontrolü
                is_drug, drug_name = self.ai.is_drug_query(test_case['input'])
                if drug_name is None:
                    raise Exception("İlaç adı tespit edilemedi")
                
                drug_match = drug_name == test_case['expected_drug_name']
                
                # Yanıt içeriği kontrolü
                content_match = self.check_content_match(response, test_case['expected_response_contains'])
                
                if emotion_match and intensity_match and drug_match and content_match:
                    self.test_results['passed'] += 1
                    status = "PASSED"
                else:
                    self.test_results['failed'] += 1
                    status = "FAILED"
                
                self.test_results['details'].append({
                    'id': test_case['id'],
                    'status': status,
                    'description': test_case['description'],
                    'emotion_match': emotion_match,
                    'intensity_match': intensity_match,
                    'drug_match': drug_match,
                    'content_match': content_match,
                    'detected_emotions': emotions,
                    'detected_intensity': intensity,
                    'response': response
                })
                
            except Exception as e:
                self.test_results['failed'] += 1
                self.test_results['details'].append({
                    'id': test_case['id'],
                    'status': "ERROR",
                    'description': test_case['description'],
                    'error': str(e)
                })

    def run_emotion_tests(self):
        """Duygu analizi testlerini çalıştırır."""
        test_cases = self.load_test_cases('test_data/emotion_scenarios.json')
        for test_case in test_cases:
            self.test_results['total'] += 1
            try:
                response = self.ai.get_response(test_case['input'])
                if response is None:
                    raise Exception("API yanıt vermedi")
                
                # Duygu ve yoğunluk kontrolü
                emotions, intensity = self.ai.detect_emotion_and_intensity(test_case['input'])
                if emotions is None or intensity is None:
                    raise Exception("Duygu analizi başarısız")
                
                emotion_match = all(e in emotions for e in test_case['expected_emotions'])
                intensity_match = abs(intensity - test_case['expected_intensity']) <= 1
                
                # Yanıt içeriği kontrolü
                content_match = self.check_content_match(response, test_case['expected_response_contains'])
                
                if emotion_match and intensity_match and content_match:
                    self.test_results['passed'] += 1
                    status = "PASSED"
                else:
                    self.test_results['failed'] += 1
                    status = "FAILED"
                
                self.test_results['details'].append({
                    'id': test_case['id'],
                    'status': status,
                    'description': test_case['description'],
                    'emotion_match': emotion_match,
                    'intensity_match': intensity_match,
                    'content_match': content_match,
                    'detected_emotions': emotions,
                    'detected_intensity': intensity,
                    'response': response
                })
                
            except Exception as e:
                self.test_results['failed'] += 1
                self.test_results['details'].append({
                    'id': test_case['id'],
                    'status': "ERROR",
                    'description': test_case['description'],
                    'error': str(e)
                })

    def run_context_tests(self):
        """Bağlam takibi testlerini çalıştırır."""
        test_cases = self.load_test_cases('test_data/context_tracking.json')
        for test_case in test_cases:
            self.test_results['total'] += 1
            try:
                # Son kullanıcı mesajını al
                last_user_message = test_case['conversation'][-1]['content']
                response = self.ai.get_response(last_user_message)
                if response is None:
                    raise Exception("API yanıt vermedi")
                
                # Yanıt içeriği kontrolü
                content_match = self.check_content_match(response, test_case['expected_response_contains'])
                
                if content_match:
                    self.test_results['passed'] += 1
                    status = "PASSED"
                else:
                    self.test_results['failed'] += 1
                    status = "FAILED"
                
                self.test_results['details'].append({
                    'id': test_case['id'],
                    'status': status,
                    'description': test_case['description'],
                    'content_match': content_match,
                    'response': response
                })
                
            except Exception as e:
                self.test_results['failed'] += 1
                self.test_results['details'].append({
                    'id': test_case['id'],
                    'status': "ERROR",
                    'description': test_case['description'],
                    'error': str(e)
                })

    def run_all_tests(self):
        """Tüm testleri çalıştırır ve raporları oluşturur."""
        print("İlaç sorguları testleri çalıştırılıyor...")
        self.run_drug_tests()
        
        print("\nDuygu analizi testleri çalıştırılıyor...")
        self.run_emotion_tests()
        
        print("\nBağlam takibi testleri çalıştırılıyor...")
        self.run_context_tests()
        
        # Test sonuçlarını kaydet
        json_file = self.save_test_results()
        html_file = self.generate_html_report()
        
        print("\n=== Test Sonuçları ===")
        print(f"Toplam Test: {self.test_results['total']}")
        print(f"Başarılı: {self.test_results['passed']}")
        print(f"Başarısız: {self.test_results['failed']}")
        print(f"\nDetaylı raporlar kaydedildi:")
        print(f"JSON Rapor: {json_file}")
        print(f"HTML Rapor: {html_file}")

if __name__ == "__main__":
    # OpenAI API anahtarını .env dosyasından al
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        print("Lütfen .env dosyasında OPENAI_API_KEY değişkenini ayarlayın")
        print("Örnek: OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    try:
        runner = TestRunner(API_KEY)
        runner.run_all_tests()
    except Exception as e:
        print(f"Hata: {str(e)}")
        sys.exit(1) 