import os
from Ilac import Drug
from ai import AIAssistant
from typing import Dict, Optional, List
from google_research.google_search import GoogleSearch


def get_openai_api_key() -> str:
    """
    OpenAI API anahtarını ortam değişkeninden alır.

    Returns:
        str: OpenAI API anahtarı.
        Raises:
            ValueError: Eğer OPENAI_API_KEY ortam değişkeni bulunamazsa.
    """
    api_key: str = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API anahtarı bulunamadı. Lütfen OPENAI_API_KEY ortam değişkenini ayarlayın.")
    return api_key


def get_google_api_keys() -> tuple[str, str]:
    """
    Google API anahtarlarını ortam değişkenlerinden alır.

    Returns:
        tuple[str, str]: Google API anahtarı ve CSE ID.
        Raises:
            ValueError: Eğer GOOGLE_API_KEY veya GOOGLE_CSE_ID ortam değişkenleri bulunamazsa.
    """
    google_api_key: str = os.environ.get("GOOGLE_API_KEY")
    cse_id: str = os.environ.get("GOOGLE_CSE_ID")

    if not google_api_key:
        raise ValueError("Google API anahtarı bulunamadı. Lütfen GOOGLE_API_KEY ortam değişkenini ayarlayın.")
    if not cse_id:
        raise ValueError("Google CSE ID bulunamadı. Lütfen GOOGLE_CSE_ID ortam değişkenini ayarlayın.")

    return google_api_key, cse_id


def get_drug_and_question() -> tuple[str, str]:
    """
    Kullanıcıdan ilaç adını ve soruyu alır.

    Returns:
        tuple[str, str]: İngilizce ilaç adı ve Türkçe soru.
    """
    drug_name_en: str = input("Lütfen ilacın İngilizce adını veya etken maddesini girin (çıkmak için 'çıkış'): ")
    if drug_name_en.lower() == 'çıkış':
        exit()  # Programı sonlandır
    user_question_tr: str = input("Lütfen sorunuzu Türkçe olarak girin: ")
    return drug_name_en, user_question_tr


def main():
    """
    Ana uygulama akışı.
    """
    try:
        api_key_openai: str = get_openai_api_key()
        ai_assistant: AIAssistant = AIAssistant(api_key_openai)

        api_key_google, cse_id_google = get_google_api_keys()
        google_search: GoogleSearch = GoogleSearch(api_key_google, cse_id_google)

        while True:
            drug_name_en, user_question_tr = get_drug_and_question()

            drug: Drug = Drug(drug_name_en)

            if drug.get_fda_info():
                drug.clean_data()

                drug_info: Dict[str, Optional[str]] = {
                    "substance_name": drug.substance_name if drug.substance_name is not None else "",
                    "indications_and_usage": drug.indications_and_usage if drug.indications_and_usage is not None else "",
                    "warnings": drug.warnings if drug.warnings is not None else "",
                    "dosage_and_administration": drug.dosage_and_administration if drug.dosage_and_administration is not None else "",
                    "adverse_reactions": drug.adverse_reactions if drug.adverse_reactions is not None else ""
                }

                prompt: str = AIAssistant.create_prompt(drug_info, user_question_tr)
                response_fda: Optional[str] = ai_assistant.get_response(prompt)

                print("\n--- FDA CEVABI ---")
                if response_fda:
                    print(response_fda)
                else:
                    print("FDA verilerinden cevap alınamadı.")

            else:
                print(f"FDA'da '{drug_name_en}' için bilgi bulunamadı. Web'de aranıyor...")
                try:
                    # drug_name parametresini ekliyoruz
                    search_queries: List[str] = google_search.create_search_queries(user_question_tr, drug_name_en)
                    all_results: List[Dict] = []
                    for query in search_queries:
                        results: List[Dict] = google_search.search_web(query)
                        all_results.extend(results)

                    if all_results:
                        print("\n--- HAM ARAMA SONUÇLARI ---")
                        for result in all_results:
                            print(f"Başlık: {result['title']}")
                            print(f"URL: {result['url']}")
                            print(f"Özet: {result['snippet']}")
                            print("-" * 20)

                        # drug_name parametresini ekliyoruz
                        answer_google: Optional[str] = google_search.analyze_and_summarize(user_question_tr, all_results, drug_name_en)
                        print("\n--- WEB ARAMA CEVABI ---")
                        if answer_google:
                            print(answer_google)
                        else:
                            print("Web arama sonuçlarından cevap oluşturulamadı.")
                    else:
                        print("\nÜzgünüm, web'de arama sonuçları bulunamadı.")
                except Exception as e:
                    print(f"Web arama sırasında hata oluştu: {e}")

    except ValueError as ve:
        print(f"HATA: {ve}")
    except Exception as e:
        print(f"Bilinmeyen bir hata oluştu: {e}")


if __name__ == "__main__":
    main()