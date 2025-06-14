# AI Sağlık Asistanı 🤖

Bu proje, ilaç bilgisi ve sağlık danışmanlığı sağlayan, empatik ve interaktif bir yapay zeka asistanıdır.

## 🌟 Özellikler

- **Empatik İletişim**: Kullanıcıların duygusal durumlarını anlayan ve psikolojik destek sağlayan yapay zeka
- **İlaç Bilgisi**: FDA verilerine dayalı güvenilir ilaç bilgileri
- **Duygu Analizi**: Kullanıcıların duygusal durumlarını tespit eden ve uygun yanıtlar veren sistem
- **İnteraktif Diyalog**: Doğal ve akıcı konuşma akışı
- **Psikolojik Destek**: Endişe ve korkuları anlayan, destekleyici yanıtlar

## 🚀 Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. `.env.example` dosyasını `.env` olarak kopyalayın ve API anahtarlarınızı ekleyin:
```bash
cp .env.example .env
```

3. Uygulamayı başlatın:
```bash
python app.py
```

## 💡 Kullanım

Asistan şu konularda yardımcı olabilir:
- İlaç bilgisi ve yan etkileri
- Sağlık danışmanlığı
- Duygusal destek
- Pratik sağlık önerileri

## 🧪 Test

Testleri çalıştırmak için:
```bash
python test_data/test_runner.py
```

Test raporları `test_reports` klasöründe oluşturulur:
- JSON formatında detaylı sonuçlar
- HTML formatında görsel raporlar

## 📁 Proje Yapısı

```
ai_module/
├── ai.py                 # Ana AI sınıfı ve prompt sistemi
├── Ilac.py              # FDA veri çekme ve temizleme
├── main.py              # Ana çalışma akışı
├── requirements.txt     # Gerekli kütüphaneler
├── test_data/          # Test senaryoları
│   ├── drug_queries.json
│   ├── emotion_scenarios.json
│   └── context_tracking.json
└── test_reports/       # Test raporları
```

## 🔧 Teknik Detaylar

- **AI Modeli**: OpenAI GPT-3.5 Turbo
- **Duygu Analizi**: Özel geliştirilmiş duygu tespit sistemi
- **Veri Kaynağı**: FDA API ve güvenilir tıbbi kaynaklar
- **Dil Desteği**: Türkçe

## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Bir Pull Request oluşturun

## ⚠️ Güvenlik Notları

- API anahtarlarınızı asla GitHub'a yüklemeyin
- `.env` dosyasını `.gitignore`'a eklediğinizden emin olun
- Test raporları hassas bilgiler içerebilir, bunları da yüklemeyin

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Geliştirici

Bu modül, projenin **Yapay Zeka** bileşeninden sorumlu [@berkayhsrt](https://github.com/berkay123001) tarafından geliştirilmiştir.

## 📅 Geliştirme Günlüğü

• [Bu Hafta Yapılanlar - 23 Mart](devlogs/BuHaftaYapılanlar_23Mart.txt)  
• [Gelecek Geliştirme Alanları](devlogs/GelecekGelistirmeAlanlari.txt)

