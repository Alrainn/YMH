import unittest
import os
import shutil
import requests # HTTP istekleri için
import json
from dotenv import load_dotenv
import time # Gecikme eklemek için
import subprocess
import signal
import atexit
import importlib

# Flask uygulamasının çalıştığı temel URL
BASE_URL = "http://127.0.0.1:5000"
ENV_FILE = '.env'
ENV_BACKUP_FILE = '.env.backup'
FLASK_PROCESS = None

def start_flask_app():
    """Flask uygulamasını başlatır"""
    global FLASK_PROCESS
    if FLASK_PROCESS is None:
        print("\n🔄 Flask uygulaması başlatılıyor...")
        FLASK_PROCESS = subprocess.Popen(['python', 'app.py'], 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        time.sleep(5)  # Flask'ın başlaması için bekle
        print("✅ Flask uygulaması başlatıldı")

def stop_flask_app():
    """Flask uygulamasını durdurur"""
    global FLASK_PROCESS
    if FLASK_PROCESS is not None:
        print("\n🛑 Flask uygulaması durduruluyor...")
        FLASK_PROCESS.terminate()
        FLASK_PROCESS.wait()
        FLASK_PROCESS = None
        print("✅ Flask uygulaması durduruldu")

# Program sonlandığında Flask'ı durdur
atexit.register(stop_flask_app)

# ÖNEMLİ: Bu test betiğini çalıştırmadan önce,
# Flask uygulamanızın (app.py) başka bir terminalde çalıştığından emin olun.
# `python app.py` komutu ile başlatabilirsiniz.

class TestAppErrorHandling(unittest.TestCase):

    original_env_vars = {}
    original_env_file_content = None

    @classmethod
    def setUpClass(cls):
        print("\n" + "="*50)
        print("AI Modülü Flask Uygulaması Hata Yönetimi Testleri Başlıyor")
        print("="*50)
        print(f"Flask uygulamasının {BASE_URL} adresinde AYRI BİR TERMİNALDE çalıştığından emin olun.")
        print("Testler sırasında .env dosyasında değişiklikler yapılacak ve")
        print("her değişiklikten sonra Flask uygulamasını (app.py) YENİDEN BAŞLATMANIZ istenecektir.")
        print("Hazırsanız Enter'a basarak testlere başlayın...")
        input()

        # Flask uygulamasını başlat
        start_flask_app()

        # Orijinal .env dosyasını yedekle
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, 'r') as f:
                cls.original_env_file_content = f.read()
            shutil.copy(ENV_FILE, ENV_BACKUP_FILE)

        # Test sırasında değiştirilebilecek ortam değişkenlerini yedekle
        keys_to_backup = ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_CSE_ID']
        for key in keys_to_backup:
            cls.original_env_vars[key] = os.environ.get(key)

    @classmethod
    def tearDownClass(cls):
        # Flask uygulamasını durdur
        stop_flask_app()

        # Orijinal .env dosyasını geri yükle
        if os.path.exists(ENV_BACKUP_FILE):
            shutil.copy(ENV_BACKUP_FILE, ENV_FILE)
            os.remove(ENV_BACKUP_FILE)
            print(f"\n{ENV_FILE} dosyası orijinal haline geri yüklendi.")
        elif cls.original_env_file_content is None and os.path.exists(ENV_FILE):
            # Testler sırasında .env oluşturulduysa ve orijinali yoksa sil
            os.remove(ENV_FILE)
            print(f"\nTestler sırasında oluşturulan {ENV_FILE} dosyası silindi.")

        print("\n" + "="*50)
        print("Tüm Testler Tamamlandı.")
        print("Lütfen Flask uygulamanızın (app.py) .env dosyasının son halini okuması için")
        print("gerekirse yeniden başlattığınızdan emin olun.")
        print("="*50)

    def _restart_flask_app(self):
        """Flask uygulamasını yeniden başlatır"""
        stop_flask_app()
        time.sleep(2)  # Uygulamanın tamamen kapanması için bekle
        start_flask_app()

    def _make_api_request(self, retries=3, delay=3):
        last_exception = None
        for attempt in range(retries):
            try:
                return requests.post(
                    f"{BASE_URL}/api/drug-info",
                    json={"drug_name_en": "Aspirin", "question_tr": "nedir"},
                    timeout=10 # İstek zaman aşımı
                )
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                print(f"Bağlantı hatası (deneme {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    print(f"{delay} saniye sonra tekrar denenecek...")
                    time.sleep(delay)
                else:
                    print("Maksimum deneme sayısına ulaşıldı.")
            except requests.exceptions.Timeout as e:
                last_exception = e
                print(f"İstek zaman aşımına uğradı (deneme {attempt + 1}/{retries}): {e}")
                # Zaman aşımı için yeniden deneme genellikle mantıklı değildir, ancak senaryoya bağlıdır.
                # Şimdilik yeniden denemeden çıkalım.
                break
        
        if last_exception:
            self.fail(f"❌ TEST BAŞARISIZ: Flask uygulamasına ({BASE_URL}) bağlanılamadı veya istek zaman aşımına uğradı. Son hata: {last_exception}")
        return None # Eğer tüm denemeler başarısız olursa

    def test_01_required_libraries(self):
        """Gerekli kütüphanelerin yüklü olup olmadığını kontrol eder"""
        print("\n📝 TEST 1: Gerekli Kütüphanelerin Kontrolü")
        required_libraries = [
            'flask',
            'openai',
            'googleapiclient',
            'python-dotenv',
            'requests',
            'json',
            'os',
            'time'
        ]
        
        missing_libraries = []
        for lib in required_libraries:
            try:
                importlib.import_module(lib.replace('-', '_'))
                print(f"✅ {lib} kütüphanesi yüklü")
            except ImportError:
                missing_libraries.append(lib)
                print(f"❌ {lib} kütüphanesi eksik")
        
        self.assertEqual(len(missing_libraries), 0, 
            f"❌ TEST BAŞARISIZ: Bazı gerekli kütüphaneler eksik: {', '.join(missing_libraries)}")
        print("✅ TEST BAŞARILI: Tüm gerekli kütüphaneler yüklü")

    def test_02_env_file_missing(self):
        print("\n📝 TEST 2: .env Dosyası Eksik")
        if os.path.exists(ENV_FILE):
            os.remove(ENV_FILE)
            print(f"-> {ENV_FILE} dosyası silindi.")

        self._restart_flask_app()

        response = self._make_api_request()
        if response: # _make_api_request başarılı bir yanıt döndürdüyse (veya self.fail ile sonlanmadıysa)
            try:
                response_json = response.json()
                print(f"Alınan yanıt (JSON): {response_json}")
                
                # Yanıt içeriğini kontrol et
                fda_response = response_json.get('fda_response', '')
                self.assertIn("❌ HATA: .env dosyası bulunamadı", fda_response, 
                    f"Yanıt beklenen .env dosyası hatasını içermiyor: {fda_response}")
                print(f"✅ TEST BAŞARILI (test_02_env_file_missing): Beklenen hata mesajı alındı")
            except json.JSONDecodeError:
                self.fail(f"❌ TEST BAŞARISIZ (test_02_env_file_missing): Yanıt JSON formatında değil. Yanıt: {response.text}")
            except AssertionError as e:
                self.fail(f"❌ TEST BAŞARISIZ (test_02_env_file_missing): {e}")

    def test_03_openai_api_key_missing_in_env_file(self):
        print("\n📝 TEST 3: OPENAI_API_KEY Eksik (.env dosyasında)")
        if os.path.exists(ENV_BACKUP_FILE):
            shutil.copy(ENV_BACKUP_FILE, ENV_FILE)

        lines_to_write = []
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.strip().startswith("OPENAI_API_KEY="):
                    lines_to_write.append("OPENAI_API_KEY=\n") 
                elif not line.strip().startswith("#") and "=" in line: 
                    lines_to_write.append(line)
        else: 
            lines_to_write = [
                "GOOGLE_API_KEY=test_google_key\n",
                "GOOGLE_CSE_ID=test_cse_id\n",
                "OPENAI_API_KEY=\n"
            ]
        
        with open(ENV_FILE, "w") as f:
            f.writelines(lines_to_write)
        print(f"-> OPENAI_API_KEY, {ENV_FILE} dosyasından silindi veya boş bırakıldı.")

        self._restart_flask_app()

        response = self._make_api_request()
        if response:
            try:
                response_json = response.json()
                print(f"Alınan yanıt (JSON): {response_json}")
                
                # Yanıt içeriğini kontrol et
                fda_response = response_json.get('fda_response', '')
                self.assertIn("❌ HATA: API anahtarı bulunamadı", fda_response, 
                    f"Yanıt beklenen API anahtarı hatasını içermiyor: {fda_response}")
                print(f"✅ TEST BAŞARILI (test_03_openai_api_key_missing_in_env_file): Beklenen hata mesajı alındı")
            except json.JSONDecodeError:
                self.fail(f"❌ TEST BAŞARISIZ (test_03_openai_api_key_missing_in_env_file): Yanıt JSON formatında değil. Yanıt: {response.text}")
            except AssertionError as e:
                self.fail(f"❌ TEST BAŞARISIZ (test_03_openai_api_key_missing_in_env_file): {e}")

    def test_04_invalid_openai_api_key(self):
        print("\n📝 TEST 4: Geçersiz OPENAI_API_KEY (.env dosyasında)")
        if os.path.exists(ENV_BACKUP_FILE):
            shutil.copy(ENV_BACKUP_FILE, ENV_FILE)

        lines_to_write = []
        openai_key_set = False
        if os.path.exists(ENV_FILE):
            with open(ENV_FILE, "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.strip().startswith("OPENAI_API_KEY="):
                    lines_to_write.append("OPENAI_API_KEY=invalid-key\n")
                    openai_key_set = True
                elif not line.strip().startswith("#") and "=" in line:
                    lines_to_write.append(line)
        
        if not openai_key_set: 
             lines_to_write.append("OPENAI_API_KEY=invalid-key\n")
             if not any(l.startswith("GOOGLE_API_KEY=") for l in lines_to_write): lines_to_write.append("GOOGLE_API_KEY=dummy_google_key_for_test\n")
             if not any(l.startswith("GOOGLE_CSE_ID=") for l in lines_to_write): lines_to_write.append("GOOGLE_CSE_ID=dummy_cse_id_for_test\n")

        with open(ENV_FILE, "w") as f:
            f.writelines(lines_to_write)
        print(f"-> OPENAI_API_KEY, {ENV_FILE} dosyasında 'invalid-key' olarak ayarlandı.")

        self._restart_flask_app()

        response = self._make_api_request()
        if response:
            try:
                response_json = response.json()
                print(f"Alınan yanıt (JSON): {response_json}")
                
                # Yanıt içeriğini kontrol et
                fda_response = response_json.get('fda_response', '')
                web_response = response_json.get('web_response', '')
                combined_response = f"{fda_response} {web_response}".lower()
                
                self.assertIn("api anahtarı geçersiz", combined_response, 
                    f"Yanıt beklenen 'API anahtarı geçersiz' mesajını içermiyor. Yanıt: {combined_response[:300]}")
                print(f"✅ TEST BAŞARILI (test_04_invalid_openai_api_key): Yanıtta 'API anahtarı geçersiz' bulundu")
            except json.JSONDecodeError:
                self.fail(f"❌ TEST BAŞARISIZ (test_04_invalid_openai_api_key): Yanıt JSON formatında değil. Yanıt: {response.text}")
            except AssertionError as e:
                self.fail(f"❌ TEST BAŞARISIZ (test_04_invalid_openai_api_key): {e}")

if __name__ == '__main__':
    unittest.main()