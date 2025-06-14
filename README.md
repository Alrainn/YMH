# YMH AI İlaç Asistanı

Bu proje, ilaçlar hakkında bilgi veren yapay zeka destekli bir asistan uygulamasıdır. OpenAI GPT-3.5 ve Google Search API'lerini kullanarak ilaçlar hakkında güvenilir bilgiler sunar.

## 🚀 Özellikler

- İlaç bilgilerini FDA veritabanından çekme
- Google araması ile güncel bilgileri bulma
- Doğal dil işleme ile samimi yanıtlar
- Web arayüzü ve API desteği
- Çoklu dil desteği (Türkçe/İngilizce)

## 📋 Gereksinimler

- Python 3.10+
- Conda veya pip
- OpenAI API anahtarı
- Google Search API anahtarı

## 🛠️ Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/yourusername/ymh-ai-assistant.git
cd ymh-ai-assistant
```

2. Conda ortamını oluşturun:
```bash
conda env create -f environment.yml
conda activate YMH_Projesi
```

3. `.env` dosyasını oluşturun:
```env
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
GOOGLE_CSE_ID=your-google-cse-id
```

4. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## 🚀 Kullanım

1. Uygulamayı başlatın:
```bash
python app.py
```

2. Web arayüzüne erişin:
```
http://localhost:5000
```

3. API kullanımı:
```bash
# İlaç bilgisi sorgulama
curl -X POST http://localhost:5000/api/drug-info \
  -H "Content-Type: application/json" \
  -d '{"drug_name_en": "aspirin", "question_tr": "Yan etkileri nelerdir?"}'

# Genel sohbet
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Merhaba, nasılsın?"}'
```

## 📁 Proje Yapısı

```
ymh-ai-assistant/
├── app.py              # Ana uygulama
├── ai.py              # AI asistan sınıfı
├── Ilac.py            # İlaç bilgisi sınıfı
├── google_research/   # Google arama fonksiyonları
├── templates/         # Web arayüzü şablonları
├── requirements.txt   # Python bağımlılıkları
└── environment.yml    # Conda ortam yapılandırması
```

## 🔧 Geliştirme

1. Yeni özellik eklemek için:
   - Yeni bir branch oluşturun
   - Değişikliklerinizi yapın
   - Test edin
   - Pull request açın

2. Hata ayıklama:
   - `DEBUG=1` environment variable'ı ile detaylı hata mesajları
   - Log dosyalarını kontrol edin

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Fork'layın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın


