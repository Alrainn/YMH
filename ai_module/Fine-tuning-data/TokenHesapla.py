import json
import tiktoken
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

def count_tokens(text: str) -> int:
    """
    Verilen metindeki token sayısını hesaplar.
    """
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 ve GPT-4 için encoding
    return len(encoding.encode(text))

def main():
    # Betiğin bulunduğu dizini al
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # d7.jsonl dosyasının betiğe göre doğru yolunu oluştur
    file_path = os.path.join(script_directory, "Cevap_Yapilandirma", "d7.jsonl")
    
    # Model adını .env'den al
    model_name = "ft:gpt-3.5-turbo-1106:personal:bioworks-gpt3-empati-009:Be8kXKjT"
    
    # Dosyanın varlığını kontrol et
    if not os.path.exists(file_path):
        print(f"HATA: Belirtilen '{file_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")
        return

    total_tokens = 0
    total_messages = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    messages = data.get('messages', [])
                    
                    for message in messages:
                        if 'content' in message:
                            total_tokens += count_tokens(message['content'])
                    
                    total_messages += 1
                    
                except json.JSONDecodeError:
                    print(f"UYARI: Geçersiz JSON satırı atlandı: {line.strip()}")
                    continue
        
        print(f"\nToken Sayımı Sonuçları:")
        print(f"Model: {model_name}")
        print(f"Toplam Token Sayısı: {total_tokens}")
        print(f"Toplam Mesaj Sayısı: {total_messages}")
        print(f"Ortalama Token/Mesaj: {total_tokens/total_messages if total_messages > 0 else 0:.2f}")
        
        # Fiyat hesaplaması
        print("\nFiyat Hesaplaması:")
        # GPT-3.5 Turbo fine-tuning fiyatları (1000 token başına)
        training_cost_per_1k = 0.0080  # $0.0080 / 1K tokens
        inference_cost_per_1k = 0.0120  # $0.0120 / 1K tokens
        
        # Örnek epoch sayısı
        epochs = 4 

        # Eğitim maliyeti
        training_cost = (total_tokens / 1000) * training_cost_per_1k * epochs
        print(f"Eğitim Maliyeti ({epochs} epoch için): ${training_cost:.4f}")
        
        # Çıktı maliyeti (örnek olarak her mesaj için 100 token çıktı varsayalım)
        output_tokens_per_message = 100
        total_output_tokens = total_messages * output_tokens_per_message
        inference_cost = (total_output_tokens / 1000) * inference_cost_per_1k
        print(f"Çıktı Maliyeti (tahmini): ${inference_cost:.4f}")
        
        # Toplam maliyet
        total_cost = training_cost + inference_cost
        print(f"Toplam Tahmini Maliyet: ${total_cost:.4f}")
        print("\nNOT: Bu fiyatlar tahminidir. Güncel fiyatlar için OpenAI'nin resmi fiyatlandırma sayfasını kontrol edin.")
        
    except Exception as e:
        print(f"HATA: Dosya işlenirken bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    main()