from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from typing import Optional, Dict, List, Tuple
import re
from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

class AIAssistant:
    """
    OpenAI API'si ile etkileşim kurarak yapay zeka asistanının işlevselliğini sağlar.
    Gelişmiş duygusal zeka ve empati yetenekleri eklenmiştir.
    """

    def __init__(self, api_key: str = None, model_name: str = "gpt-3.5-turbo"):
        """
        AIAssistant sınıfının yapıcısı.
        Optimize edilmiş versiyon.

        Args:
            api_key (str, optional): OpenAI API anahtarı. Belirtilmezse .env dosyasından okunur.
            model_name (str, optional): Kullanılacak OpenAI modeli.
                Varsayılan: "gpt-3.5-turbo".
        """
        self.client: OpenAI = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name: str = model_name if model_name != "gpt-3.5-turbo-ft" else os.getenv("FINE_TUNED_MODEL", "ft:gpt-3.5-turbo-1106:personal:bioworks-gpt3-empati-006:BdOcBDoD")
        
        # İlaç arama optimizasyonu için sabitler
        self.drug_name_variations = {
            "gripin": ["gripin", "gripin ilaç", "gripin tablet", "gripin şurup"],
            "deloday": ["deloday", "deloday tablet", "deloday şurup"],
            # Diğer ilaçlar için varyasyonlar eklenebilir
        }
        
        self.drug_info_sources = [
            "TİTCK (Türkiye İlaç ve Tıbbi Cihaz Kurumu)",
            "Resmi ilaç prospektüsleri",
            "Güvenilir tıbbi kaynaklar"
        ]
        
        # Duygu tespiti için optimize edilmiş pattern'ler
        self.emotion_patterns: Dict[str, List[str]] = {
            "endişe": ["endişe", "korku", "tedirgin", "risk", "tehlike", "yan etki", "korkuyorum", "şüphe", "güvensiz"],
            "acı": ["ağrı", "sızı", "acı", "rahatsızlık", "hasta", "kötü hissediyorum", "fenalık"],
            "umut": ["iyileşme", "tedavi", "düzelme", "fayda", "iyi", "umut", "olumlu"],
            "belirsizlik": ["emin değilim", "bilmiyorum", "acaba", "mı", "mi", "kararsız", "şüpheliyim"],
            "depresif": ["üzgün", "mutsuz", "depresyon", "bunalım", "sıkıntı", "yalnız", "çaresiz"],
            "kaygı": ["obsesif", "kompulsif", "takıntı", "panik", "anksiyete", "gergin", "stres"],
            "moral": ["moral", "motivasyon", "destek", "güven", "cesaret", "güçlü", "pozitif"]
        }
        
        # Empati prefix'leri - optimize edilmiş
        self.empathy_prefixes = {
            "endişe": [
                "Endişelenmeniz çok doğal, birlikte bu durumu değerlendirelim. ",
                "Endişelerinizi anlıyorum ve size yardımcı olmak istiyorum. ",
                "Bu konudaki endişelerinizi benimle paylaştığınız için teşekkür ederim. ",
                "Endişelerinizi ciddiye alıyorum ve size destek olmak için buradayım. ",
                "Bu durumda endişelenmek normal, ancak birlikte çözüm bulabiliriz. "
            ],
            "acı": [
                "Bu durumun sizin için zor olduğunu anlıyorum. ",
                "Yaşadığınız rahatsızlığın farkındayım ve size yardımcı olmak istiyorum. ",
                "Bu zor süreçte yanınızda olduğumu bilmenizi isterim. ",
                "Yaşadığınız sıkıntıyı anlıyorum ve size destek olmak için buradayım. ",
                "Bu zorlu durumla başa çıkmaya çalışırken yanınızdayım. "
            ]
        }
        
        # Destekleyici cümleler - optimize edilmiş
        self.supportive_endings = [
            "Unutmayın, yanınızdayım ve size destek olmak için buradayım.",
            "Endişelerinizi paylaşabildiğiniz için teşekkür ederim. Size yardımcı olmaya devam edeceğim.",
            "Bu süreci birlikte yönetebiliriz. Her zaman sorularınızı sorabilirsiniz.",
            "Sizi dinliyorum ve anlıyorum. Birlikte en iyi çözümü bulacağız.",
            "Her zaman yanınızdayım ve size destek olmaktan mutluluk duyuyorum."
        ]
        
        # Yan etki soruları - optimize edilmiş
        self.side_effect_questions = [
            "Şu an kendinizi nasıl hissediyorsunuz?",
            "İlaçla ilgili spesifik bir endişeniz var mı?",
            "Size yardımcı olabilecek başka bilgiler verebilir miyim?",
            "Daha önce benzer ilaçlar kullandınız mı?",
            "Bu endişelerinizi doktorunuzla paylaştınız mı?"
        ]
        
        # Pratik öneriler - optimize edilmiş
        self.practical_tips = [
            "\n\nBazı pratik öneriler:",
            "- İlacı düzenli saatlerde kullanmak yan etki riskini azaltabilir",
            "- İlacı tok karnına almak mide rahatsızlığı riskini düşürür",
            "- Herhangi bir yan etki hissettiğinizde not almanız faydalı olabilir",
            "- Düzenli su içmek ve dengeli beslenmek yan etkileri azaltabilir",
            "- Stres yönetimi ve düzenli uyku da önemlidir"
        ]
        
        # İlaç takip soruları - optimize edilmiş
        self.drug_follow_up_questions = [
            "Bu ilacı daha önce kullandınız mı?",
            "Şu ana kadar herhangi bir rahatsızlık hissettiniz mi?",
            "İlacı yemeklerle birlikte mi alıyorsunuz?",
            "Günün hangi saatlerinde kullanıyorsunuz?",
            "Doktorunuz size ilacın kullanımıyla ilgili özel öneriler verdi mi?"
        ]
        
        # Takip soruları - optimize edilmiş
        self.follow_up_questions = {
            "endişe": [
                "Endişenizin kaynağı nedir? Bunu benimle paylaşmak ister misiniz?",
                "Bu konuda sizi en çok endişelendiren şey nedir?",
                "Endişelerinizi azaltmak için size nasıl yardımcı olabilirim?",
                "Bu endişeyi ne zamandır yaşıyorsunuz?",
                "Daha önce benzer endişeler yaşadınız mı?"
            ],
            "acı": [
                "Rahatsızlığınızı biraz daha açıklar mısınız?",
                "Bu belirtileri ne zamandır yaşıyorsunuz?",
                "Şikayetleriniz günün hangi saatlerinde artıyor?",
                "Ağrınızı hafifletmek için neler deniyorsunuz?",
                "Bu durumu doktorunuzla paylaştınız mı?"
            ]
        }

    @staticmethod
    def _format_section(title: str, content: Optional[str]) -> str:
        """
        Prompt'a eklenecek bir bölümü biçimlendirir.

        Args:
            title (str): Bölüm başlığı.
            content (Optional[str]): Bölüm içeriği.

        Returns:
            str: Biçimlendirilmiş bölüm.
        """
        if content:
            return f"{title}:\n{content}\n"
        return ""

    @staticmethod
    def create_prompt(drug_info: Dict, question: str) -> str:
        """
        FDA verisine dayalı olarak GPT-3.5 için kapsamlı bir prompt oluşturur.

        Args:
            drug_info (Dict): FDA'dan çekilen ilaç bilgileri.
            question (str): Kullanıcının Türkçe sorusu.

        Returns:
            str: Oluşturulan prompt.
        """
        substance: str = drug_info.get('substance_name', 'UNKNOWN')
        prompt_start: str = f"Aşağıdaki bilgiler, {substance} etken maddeli ilaç için FDA etiketinden alınmıştır.\n\n"
        prompt_sections: List[str] = [prompt_start]

        # Bölümleri ekle
        prompt_sections.append(
            AIAssistant._format_section(
                "ENDİKASYONLAR VE KULLANIM", drug_info.get("indications_and_usage")
            )
        )
        prompt_sections.append(
            AIAssistant._format_section(
                "DOZAJ VE UYGULAMA", drug_info.get("dosage_and_administration")
            )
        )
        prompt_sections.append(
            AIAssistant._format_section("UYARILAR", drug_info.get("warnings"))
        )

        # Yan etkiler
        adverse: Optional[str] = drug_info.get("adverse_reactions")
        if not adverse and "yan etki" in question.lower():
            prompt_sections.append(
                "YAN ETKİLER:\nYan etkilerle ilgili bilgi ayrı olarak listelenmemiş olabilir, ancak yukarıdaki UYARILAR bölümünde bulunabilir.\n"
            )
        elif adverse:
            prompt_sections.append(f"YAN ETKİLER:\n{adverse}\n")

        # Soru türüne göre prompt
        if "nedir" in question.lower() and "ne için kullanılır" in question.lower():
            prompt_sections.append(
                "Lütfen aşağıdaki soruyu YUKARIDAKİ FDA verilerine dayanarak, Türkçe ve öz bir şekilde yanıtlayın.\n"
                "Sadece ilacın *ne olduğu* ve *ne için kullanıldığına* dair bilgi verin. Başka hiçbir bilgi eklemeyin.\n"
                "Cevabınızı madde işaretleri kullanmadan, kısa ve net cümlelerle yazın.\n"
                "Örnek Cevap:\n[İlaç adı], [ilaç sınıfı] bir ilaçtır. [Hastalık/durum] tedavisinde kullanılır.\n"
                "Eğer FDA etiketinde bu soruların cevabı *doğrudan* verilmiyorsa, cevap vermeyin.\n"
            )
        elif question.lower() == "yan etkileri nelerdir":  # Tam eşleşme
            prompt_sections.append(
                "Lütfen aşağıdaki soruyu YUKARIDAKİ FDA verilerine dayanarak, Türkçe ve kapsamlı biçimde yanıtlayın.\n"
                "Sadece ilacın *yan etkilerini* listeleyin. Başka hiçbir bilgi eklemeyin.\n"
                "Cevabınızı madde işaretleri kullanarak listeleyin ve her bir yan etkinin ne kadar yaygın olduğunu belirtin.\n"
                "Örnek Cevap:\nÇok yaygın görülen yan etkiler (örneğin, %10'dan fazla hastada görülür):\n- Baş ağrısı (%X oranında görülür): [Açıklama]\n- Mide bulantısı (%Y oranında görülür): [Açıklama]\nYaygın görülen yan etkiler (örneğin, %1 ila %10 arasında hastada görülür):\n- İshal: [Açıklama]\n- ...\nEğer FDA etiketinde bu soruların cevabı *doğrudan* verilmiyorsa, cevap vermeyin.\n"
            )
        elif "ve" in question.lower():  # "ve" bağlacı varsa iki ayrı soruya cevap ver
            prompt_sections.append(
                "Lütfen aşağıdaki soruyu iki ayrı soru olarak değerlendirin ve her birine YUKARIDAKİ FDA verilerine dayanarak, Türkçe ve kapsamlı biçimde yanıtlayın.\n"
                "Soru 1: [İlk soru]\nCevap 1:\n- [Cevap 1.1]\n- [Cevap 1.2]\nKaynak: [İlgili FDA bölümü]\n\nSoru 2: [İkinci soru]\nCevap 2:\n- [Cevap 2.1]\n- [Cevap 2.2]\nKaynak: [İlgili FDA bölümü]\n"
                "Eğer FDA etiketinde sorunun cevabı *doğrudan* verilmiyorsa, o soruya cevap vermeyin.\n"
            )
        else:  # Diğer sorular için genel prompt
            prompt_sections.append(
                "Aşağıdaki Türkçe soruyu, mümkün olduğunca YUKARIDAKİ FDA verilerine dayanarak, Türkçe ve kapsamlı biçimde yanıtlayın.\n"
                "Cevabınızı madde işaretleri kullanarak listeleyin.\n"
                "Eğer sorunun yanıtı doğrudan ve açıkça verilmiyorsa, ilgili bölümleri yorumlayarak genel bilgi sunmaya çalışın.\n"
                "Yine de emin değilseniz, 'FDA verilerinde bu konuda net bir bilgi bulunmamaktadır.' diyebilirsiniz. Ancak bu ifadeyi sadece FDA etiketinde sorunun cevabı ile ilgili *hiçbir* bilgi yoksa kullanın.\n"
            )

        prompt_sections.append(f"Soru: {question}\n\nCevap:")
        fda_link: str = f"https://labels.fda.gov/#search/{substance}"
        prompt_sections.append(f"\nKaynak: {fda_link}")  # Kaynak belirtme

        return "\n".join(prompt_sections)

    def detect_emotion_and_intensity(self, text: str) -> Tuple[List[str], int]:
        """
        Metindeki duygu durumlarını ve yoğunluğunu tespit eder.
        Optimize edilmiş versiyon.

        Args:
            text (str): Analiz edilecek metin.

        Returns:
            Tuple[List[str], int]: Tespit edilen duygular ve yoğunluk seviyesi (1-5).
        """
        detected_emotions = []
        lower_text = text.lower()
        intensity = 1
        
        # Duygu tespiti - optimize edilmiş
        for emotion, patterns in self.emotion_patterns.items():
            if any(pattern in lower_text for pattern in patterns):
                detected_emotions.append(emotion)
                # Yoğunluk analizi - optimize edilmiş
                intensity = min(5, intensity + sum(1 for pattern in patterns if pattern in lower_text))
        
        # Ünlem işaretleri ve tekrarlanan karakterler - optimize edilmiş
        intensity = min(5, intensity + text.count("!"))
        if any(char * 2 in text for char in "!?.,"):
            intensity = min(5, intensity + 1)
            
        return detected_emotions, intensity

    def create_empathetic_response(self, response: str, emotions: List[str], intensity: int) -> str:
        """
        Yanıtı duygusal bağlam ve yoğunluğa göre şekillendirir.
        Optimize edilmiş versiyon.

        Args:
            response (str): Orijinal yanıt.
            emotions (List[str]): Tespit edilen duygular.
            intensity (int): Duygu yoğunluğu (1-5).

        Returns:
            str: Empatik yanıt.
        """
        if not emotions:
            return response

        # İlaçla ilgili anahtar kelimeleri kontrol et
        drug_related = any(keyword in response.lower() for keyword in ["ilaç", "tablet", "kapsül", "yan etki", "doz", "tedavi"])
        has_side_effects = "yan etki" in response.lower()
        
        # Duygu bazlı prefix seçimi
        emotion = emotions[0]
        prefix = self.empathy_prefixes.get(emotion, ["Size yardımcı olmak istiyorum. "])[min(intensity - 1, len(self.empathy_prefixes.get(emotion, ["Size yardımcı olmak istiyorum. "])) - 1)]
        
        # Destekleyici cümle seçimi
        ending = self.supportive_endings[min(intensity - 1, len(self.supportive_endings) - 1)]
        
        # İnteraktif sorular ve öneriler
        interactive_questions = []
        
        if drug_related and has_side_effects:
            interactive_questions.extend([
                "Hangi yan etkilerden endişeleniyorsunuz?",
                "Bu endişelerinizin kaynağı nedir?",
                "Daha önce benzer ilaçlar kullandınız mı?",
                "Bu endişelerinizi doktorunuzla paylaştınız mı?",
                "Yan etkiler günlük hayatınızı nasıl etkiliyor?",
                "Bu konuda size nasıl destek olabilirim?",
                "Yan etkilerle başa çıkmak için neler deniyorsunuz?",
                "Doktorunuz size yan etkiler hakkında bilgi verdi mi?"
            ])
        elif drug_related:
            interactive_questions.extend([
                "Bu ilaç hakkında özellikle endişelendiğiniz bir konu var mı?",
                "İlacı kullanmaya başladıktan sonra nasıl hissediyorsunuz?",
                "Doktorunuz size ilacın kullanımıyla ilgili özel öneriler verdi mi?",
                "İlaç hakkında başka merak ettiğiniz bir şey var mı?",
                "İlacın etkisini görmeye başladınız mı?",
                "İlaç kullanımıyla ilgili endişeleriniz var mı?",
                "Size başka nasıl yardımcı olabilirim?",
                "Bu ilaç hakkında başka sorularınız var mı?"
            ])
        elif emotion in self.follow_up_questions:
            interactive_questions.extend([
                "Bu endişenin kaynağı nedir?",
                "Bu durum sizi nasıl etkiliyor?",
                "Daha önce benzer endişeler yaşadınız mı?",
                "Size nasıl destek olabilirim?",
                "Bu endişelerinizi başkalarıyla paylaştınız mı?",
                "Endişelerinizle başa çıkmak için neler yapıyorsunuz?",
                "Size yardımcı olabilecek başka bir konu var mı?",
                "Bu konuda profesyonel destek almayı düşündünüz mü?"
            ])

        # Pratik öneriler
        practical_tips = []
        if drug_related:
            practical_tips.extend([
                "\n\nBazı pratik öneriler:",
                "- İlacı düzenli saatlerde kullanmak yan etki riskini azaltabilir",
                "- İlacı tok karnına almak mide rahatsızlığı riskini düşürür",
                "- Herhangi bir yan etki hissettiğinizde not almanız faydalı olabilir",
                "- Düzenli su içmek ve dengeli beslenmek yan etkileri azaltabilir",
                "- Stres yönetimi ve düzenli uyku da önemlidir",
                "- Doktorunuzla düzenli iletişim halinde olun"
            ])

        # Yanıtı oluştur
        final_response = f"{prefix}{response}\n\n"
        
        # İnteraktif soruları ekle
        if interactive_questions:
            final_response += "Biraz daha konuşmak ister misiniz?\n"
            for question in interactive_questions[:3]:  # İlk üç soruyu seç
                final_response += f"- {question}\n"
        
        # Pratik önerileri ekle
        if practical_tips:
            final_response += "\n".join(practical_tips)
        
        # Destekleyici cümleyi ekle
        final_response += f"\n\n{ending}"
        
        # Konuşmayı devam ettir
        if drug_related:
            final_response += "\n\nBu ilaç hakkında özellikle endişelendiğiniz bir konu var mı? Ya da yan etkiler konusunda endişelerinizi paylaşmak ister misiniz? Size yardımcı olmak için buradayım."
        else:
            final_response += "\n\nBu konuda başka düşünceleriniz var mı? Ya da başka bir konuda yardıma ihtiyacınız var mı? Sizi dinlemek için buradayım."
        
        return final_response

    def is_drug_query(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Metnin ilaç sorgusu olup olmadığını kontrol eder.
        Optimize edilmiş versiyon.

        Args:
            text (str): Kontrol edilecek metin.

        Returns:
            Tuple[bool, Optional[str]]: (İlaç sorgusu mu?, İlaç adı)
        """
        lower_text = text.lower()
        
        # İlaç adı varyasyonlarını kontrol et
        for drug_name, variations in self.drug_name_variations.items():
            if any(var in lower_text for var in variations):
                return True, drug_name
                
        return False, None

    def get_response(self, prompt: str) -> str:
        try:
            # Önce .env dosyasının varlığını kontrol et
            try:
                from dotenv import load_dotenv
                import os
                load_dotenv()
                
                if not os.path.exists('.env'):
                    return "❌ HATA: .env dosyası bulunamadı!\n\n" + \
                           "Lütfen şunları yapın:\n" + \
                           "1. .env.example dosyasını .env olarak kopyalayın:\n" + \
                           "   cp .env.example .env\n" + \
                           "2. .env dosyasını bir metin editörü ile açın\n" + \
                           "3. OPENAI_API_KEY değerini kendi API anahtarınızla değiştirin\n" + \
                           "4. Dosyayı kaydedin ve uygulamayı yeniden başlatın"
                
                # API anahtarının varlığını kontrol et
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    return "❌ HATA: API anahtarı bulunamadı!\n\n" + \
                           "Lütfen şunları yapın:\n" + \
                           "1. .env dosyasını açın\n" + \
                           "2. OPENAI_API_KEY= değerini kontrol edin\n" + \
                           "3. API anahtarınızı doğru formatta ekleyin (sk- ile başlamalı)\n" + \
                           "4. Dosyayı kaydedin ve uygulamayı yeniden başlatın"
                
                # Model adının varlığını kontrol et
                model_name = os.getenv('MODEL_NAME')
                if not model_name:
                    return "❌ HATA: Model adı bulunamadı!\n\n" + \
                           "Lütfen şunları yapın:\n" + \
                           "1. .env dosyasını açın\n" + \
                           "2. MODEL_NAME= değerini kontrol edin\n" + \
                           "3. Model adını ekleyin (örn: gpt-3.5-turbo)\n" + \
                           "4. Dosyayı kaydedin ve uygulamayı yeniden başlatın"
                
            except ImportError:
                return "❌ HATA: python-dotenv paketi eksik!\n\n" + \
                       "Lütfen şu komutu çalıştırın:\n" + \
                       "pip install python-dotenv"

            # Gerekli modüllerin varlığını kontrol et
            try:
                import openai
            except ImportError:
                return "❌ HATA: openai paketi eksik!\n\n" + \
                       "Lütfen şu komutu çalıştırın:\n" + \
                       "pip install openai"

            try:
                emotions, intensity = self.detect_emotion_and_intensity(prompt)
                is_drug, drug_name = self.is_drug_query(prompt)
            except Exception as e:
                return f"❌ HATA: Duygu analizi sırasında bir sorun oluştu!\n\n" + \
                       f"Hata detayı: {str(e)}\n\n" + \
                       "Lütfen şunları kontrol edin:\n" + \
                       "1. Tüm gerekli paketlerin yüklendiğinden emin olun:\n" + \
                       "   pip install -r requirements.txt\n" + \
                       "2. Uygulamayı yeniden başlatın"
            
            SYSTEM_PROMPT = """Sen Türkçe konuşan, empatik ve psikolojik destek odaklı bir sağlık asistanısın. 
            Kullanıcıların duygusal durumlarını derinlemesine anlamaya çalışmalı ve onlara psikolojik destek sağlamalısın.

            TEMEL İLKELER:
            1. Aktif Dinleme ve Empati:
               - Kullanıcının duygularını yansıt
               - "Anlıyorum", "Bu gerçekten zor olmalı" gibi empatik ifadeler kullan
               - Duyguları isimlendir ve doğrula
               - Kullanıcının deneyimlerini önemse

            2. Psikolojik Destek:
               - Kullanıcının endişelerini normalleştir
               - Güven verici bir ton kullan
               - Umut aşıla
               - Yalnız olmadığını hissettir
               - Başa çıkma stratejileri öner

            3. İnteraktif Diyalog:
               - Her yanıtta en az bir açık uçlu soru sor
               - Kullanıcının yanıtlarını bekleyerek ilerle
               - Konuyu doğal bir şekilde derinleştir
               - Kullanıcıyı konuşmaya teşvik et
               - Bağlamı takip et ve referans ver

            4. İlaç Bilgisi ve Güvenlik:
               - Bilimsel ve güvenilir bilgiler ver
               - Yan etkileri dengeli bir şekilde açıkla
               - Doktora danışmanın önemini vurgula
               - Endişeleri azaltmaya çalış
               - Pratik öneriler sun

            ÖRNEK YANITLAR:

            Kullanıcı: "Çok endişeliyim"
            Sen: "Endişeli hissettiğini duyduğuma üzüldüm. Bu duyguyu benimle paylaştığın için teşekkür ederim. Endişenin seni nasıl etkilediğini anlatmak ister misin? Seni dinlemek için buradayım."

            Kullanıcı: "İlacın yan etkilerinden korkuyorum"
            Sen: "Yan etkilerden korkman çok doğal bir duygu. Bu korkunun seni nasıl etkilediğini anlatmak ister misin? Ayrıca, hangi yan etkiler seni özellikle endişelendiriyor?"

            Kullanıcı: [İlaç adı]
            Sen: "[İlaç bilgisi]... Bu ilaç hakkında özellikle merak ettiğin bir şey var mı? Yan etkiler konusunda endişelerin varsa, bunları birlikte konuşabiliriz. Seni dinlemek için buradayım."

            KÖTÜ YANITLAR (BUNLARI ASLA KULLANMA):
            - "Size nasıl yardımcı olabilirim?" (çok genel)
            - "Neden endişelisin?" (çok direkt)
            - "Başka bir sorunuz var mı?" (konuşmayı kapatıcı)
            - "Endişelenmeyin" (duyguları geçersiz kılıyor)
            - "Her şey yoluna girecek" (boş umut veriyor)

            ÖNEMLİ:
            - Her yanıtın sonunda konuşmayı devam ettir
            - Kullanıcının duygularını ciddiye al
            - Profesyonel ama samimi bir ton kullan
            - Gerektiğinde doktora yönlendir
            - Her zaman destekleyici ol"""

            system_message = SYSTEM_PROMPT

            messages: List[ChatCompletionMessageParam] = [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content=system_message,
                ),
                ChatCompletionUserMessageParam(role="user", content=prompt),
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=2500,
                    temperature=0.7,
                )
            except Exception as api_error:
                error_message = str(api_error)
                if "API key" in error_message:
                    return "❌ HATA: API anahtarı geçersiz!\n\n" + \
                           "Lütfen şunları kontrol edin:\n" + \
                           "1. .env dosyasındaki API anahtarının doğru olduğundan emin olun\n" + \
                           "2. API anahtarının sk- ile başladığından emin olun\n" + \
                           "3. API anahtarının aktif olduğundan emin olun\n" + \
                           "4. OpenAI hesabınızın aktif olduğundan emin olun\n\n" + \
                           "Eğer sorun devam ederse:\n" + \
                           "1. OpenAI hesabınıza giriş yapın\n" + \
                           "2. Yeni bir API anahtarı oluşturun\n" + \
                           "3. .env dosyasını güncelleyin"
                elif "rate limit" in error_message.lower():
                    return "❌ HATA: API kullanım limitine ulaşıldı!\n\n" + \
                           "Lütfen birkaç dakika bekleyip tekrar deneyin.\n" + \
                           "Eğer sorun devam ederse, OpenAI hesabınızdaki kullanım limitlerini kontrol edin."
                elif "model" in error_message.lower():
                    return "❌ HATA: Model bulunamadı!\n\n" + \
                           "Lütfen şunları kontrol edin:\n" + \
                           "1. .env dosyasındaki MODEL_NAME değerinin doğru olduğundan emin olun\n" + \
                           "2. Model adının gpt-3.5-turbo olduğundan emin olun\n" + \
                           "3. OpenAI hesabınızın bu modele erişim izni olduğundan emin olun"
                else:
                    return f"❌ HATA: API bağlantı hatası!\n\n" + \
                           f"Hata detayı: {error_message}\n\n" + \
                           "Lütfen şunları kontrol edin:\n" + \
                           "1. İnternet bağlantınızın olduğundan emin olun\n" + \
                           "2. Tüm gerekli paketlerin yüklendiğinden emin olun:\n" + \
                           "   pip install -r requirements.txt\n" + \
                           "3. .env dosyasının doğru yapılandırıldığından emin olun\n" + \
                           "4. Uygulamayı yeniden başlatın"
            
            try:
                raw_response = response.choices[0].message.content
                empathetic_response = self.create_empathetic_response(raw_response, emotions, intensity)
                return empathetic_response
            except Exception as e:
                return f"❌ HATA: Yanıt oluşturulurken bir sorun oluştu!\n\n" + \
                       f"Hata detayı: {str(e)}\n\n" + \
                       "Lütfen şunları kontrol edin:\n" + \
                       "1. Tüm gerekli paketlerin yüklendiğinden emin olun:\n" + \
                       "   pip install -r requirements.txt\n" + \
                       "2. Uygulamayı yeniden başlatın"
            
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            
            if "OpenAI" in error_type:
                return "❌ HATA: OpenAI bağlantı hatası!\n\n" + \
                       "Lütfen şunları kontrol edin:\n" + \
                       "1. .env dosyasının doğru konumda olduğundan emin olun\n" + \
                       "2. API anahtarınızın doğru olduğundan emin olun\n" + \
                       "3. İnternet bağlantınızı kontrol edin\n" + \
                       "4. Uygulamayı yeniden başlatın"
            
            elif "ModuleNotFoundError" in error_type:
                missing_module = error_message.split("'")[1]
                return f"❌ HATA: Eksik modül: {missing_module}\n\n" + \
                       "Lütfen şu komutu çalıştırın:\n" + \
                       "pip install -r requirements.txt\n\n" + \
                       "Eğer sorun devam ederse:\n" + \
                       "1. Python'un doğru sürümünü kullandığınızdan emin olun\n" + \
                       "2. Sanal ortam (virtual environment) kullanıyorsanız aktif olduğundan emin olun"
            
            elif "FileNotFoundError" in error_type:
                return "❌ HATA: Gerekli dosya bulunamadı!\n\n" + \
                       "Lütfen şunları kontrol edin:\n" + \
                       "1. .env dosyasının proje klasöründe olduğundan emin olun\n" + \
                       "2. Tüm dosyaların doğru konumda olduğunu kontrol edin\n" + \
                       "3. GitHub'dan tüm dosyaları indirdiğinizden emin olun\n\n" + \
                       "Eğer .env dosyası eksikse:\n" + \
                       "1. .env.example dosyasını .env olarak kopyalayın\n" + \
                       "2. .env dosyasını düzenleyin\n" + \
                       "3. Uygulamayı yeniden başlatın"
            
            else:
                return f"❌ HATA: Beklenmeyen bir hata oluştu!\n\n" + \
                       f"Hata türü: {error_type}\n" + \
                       f"Hata mesajı: {error_message}\n\n" + \
                       "Lütfen şunları yapın:\n" + \
                       "1. Uygulamayı yeniden başlatın\n" + \
                       "2. Tüm gerekli paketleri yükleyin:\n" + \
                       "   pip install -r requirements.txt\n" + \
                       "3. .env dosyasını kontrol edin\n" + \
                       "4. İnternet bağlantınızı kontrol edin\n" + \
                       "5. Sorun devam ederse, lütfen GitHub'daki Issues bölümüne bildirin" 