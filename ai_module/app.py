from flask import Flask, request, jsonify, render_template
import os
from Ilac import Drug
from ai import AIAssistant
from google_research.google_search import GoogleSearch
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps

# .env dosyasını yükle
load_dotenv()

app = Flask(__name__)

# CORS ayarları - Tüm originlere izin ver (geçici olarak)
CORS(app)

# Rate limiting ayarları
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# API key doğrulama decorator'ı - geçici olarak devre dışı
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated

# Konuşma geçmişini saklamak için basit bir bellek
conversation_memory = {}

def get_openai_api_key() -> str:
    """
    OpenAI API anahtarını ortam değişkeninden alır.
    """
    api_key: str = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API anahtarı bulunamadı. Lütfen OPENAI_API_KEY ortam değişkenini ayarlayın.")
    return api_key

def get_google_api_keys() -> tuple[str, str]:
    """
    Google API anahtarlarını ortam değişkenlerinden alır.
    """
    google_api_key: str = os.environ.get("GOOGLE_API_KEY")
    cse_id: str = os.environ.get("GOOGLE_CSE_ID")

    if not google_api_key:
        raise ValueError("Google API anahtarı bulunamadı. Lütfen GOOGLE_API_KEY ortam değişkenini ayarlayın.")
    if not cse_id:
        raise ValueError("Google CSE ID bulunamadı. Lütfen GOOGLE_CSE_ID ortam değişkenini ayarlayın.")

    return google_api_key, cse_id

def get_conversation_history(session_id: str) -> List[Dict]:
    """
    Belirli bir oturum için konuşma geçmişini döndürür.
    """
    if session_id in conversation_memory:
        # 1 saatten eski konuşmaları temizle
        current_time = datetime.now()
        conversation_memory[session_id] = [
            conv for conv in conversation_memory[session_id]
            if current_time - conv['timestamp'] < timedelta(hours=1)
        ]
        return conversation_memory[session_id]
    return []

def add_to_conversation_history(session_id: str, drug_name: str, question: str, response: str):
    """
    Konuşma geçmişine yeni bir mesaj ekler.
    """
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    
    conversation_memory[session_id].append({
        'timestamp': datetime.now(),
        'drug_name': drug_name,
        'question': question,
        'response': response
    })

def generate_natural_response(drug_name: str, raw_response: str, conversation_history: List[Dict], question: str, model_name: str = "gpt-3.5-turbo") -> str:
    """
    FDA veya Google'dan gelen ham yanıtı alıp daha doğal, sohbet tarzında bir yanıt oluşturur.
    """
    api_key: str = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    
    # Model seçimi
    if model_name == "gpt-4":
        model = "gpt-4"
    elif model_name == "gpt-3.5-turbo-ft":
        model = os.getenv("FINE_TUNED_MODEL", "ft:gpt-3.5-turbo-1106:personal:bioworks-gpt3-empati-006:BdOcBDoD")
    else:
        model = model_name
    
    # Konuşma geçmişini OpenAI formatına dönüştür
    messages = [
        {"role": "system", "content": f"""
        Sen ilaçlar hakkında bilgi veren, samimi ve yardımsever bir sağlık asistanısın.
        Kullanıcının sorduğu sorulara aşağıdaki bilgilere dayanarak cevap vereceksin.
        Cevaplarında resmi tıbbi dil yerine, daha samimi ve anlaşılır bir dil kullan.
        İlaç bilgilerini sunarken fazla teknik terimleri açıkla ve kullanıcının endişelerini dinle.
        Bilgi kaynağının FDA veya web aramaları olduğunu belirtmene gerek yok.
        Cevaplarında kısa ve öz ol, ancak önemli bilgileri atlamadan ve samimi bir dilde yanıtla.
        
        Doktor yönlendirmesi yaparken:
        - Her yanıtın sonunda otomatik olarak doktora yönlendirme yapma
        - Sadece gerçekten ciddi durumlarda (acil, hayati risk, şiddetli yan etki gibi) doktora yönlendir
        - Yönlendirme yaparken doğal bir sohbet akışı içinde yap, "doktorunuza danışın" gibi resmi ifadeler kullanma
        - Kullanıcının endişelerini önce dinle ve anla, sonra gerekirse yönlendirme yap
        
        Şu anda kullanıcı seninle {drug_name} ilacı hakkında konuşuyor.
        """}
    ]
    
    # Konuşma geçmişini ekle
    for entry in conversation_history:
        if 'role' in entry and 'content' in entry:
            messages.append({"role": entry['role'], "content": entry['content']})
    
    # Kullanıcının mevcut sorusunu ekle
    messages.append({"role": "user", "content": question})
    
    # Ham yanıtı ekle
    messages.append({"role": "system", "content": f"Aşağıdaki ham bilgiler mevcut soru için bulundu, bunu kullanıcıya samimi şekilde ilet: {raw_response}"})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=3000,
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def simple_chat_response(message: str, conversation_history: List[Dict], model_name: str = "gpt-3.5-turbo") -> str:
    """
    İlaç araması yapmadan basit sohbet yanıtları üretir.
    """
    api_key: str = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    
    # Model seçimi
    if model_name == "gpt-4":
        model = "gpt-4"
    elif model_name == "gpt-3.5-turbo-ft":
        model = os.getenv("FINE_TUNED_MODEL", "ft:gpt-3.5-turbo-1106:personal:bioworks-gpt3-empati-006:BdOcBDoD")
    else:
        model = model_name
    
    messages = [
        {"role": "system", "content": """
        Sen samimi ve yardımsever bir sağlık asistanısın. Sağlık ve ilaçlar konusunda genel bilgiler verebilirsin,
        ancak spesifik ilaç tavsiyeleri yapmamalısın. Kullanıcıyla doğal bir sohbet akışı içinde
        konuşmalısın.
        
        Eğer kullanıcı sadece bir ilaç adı yazarsa:
        - İlaç hakkında genel bir bilgi ver
        - İlacı internetten araştırmak isteyip istemediğini sor
        - Örnek: "Bu ilaç hakkında daha detaylı bilgi almak ister misiniz? İnternetten araştırabilirim."
        
        Yanıtlarını kısa ve anlaşılır tut, tıbbi terimleri kullanmaktan kaçın.
        
        Doktor yönlendirmesi yaparken:
        - Her yanıtın sonunda otomatik olarak doktora yönlendirme yapma
        - Sadece gerçekten ciddi durumlarda (acil, hayati risk, şiddetli yan etki gibi) doktora yönlendir
        - Yönlendirme yaparken doğal bir sohbet akışı içinde yap, "doktorunuza danışın" gibi resmi ifadeler kullanma
        - Kullanıcının endişelerini önce dinle ve anla, sonra gerekirse yönlendirme yap
        """}
    ]
    
    # Konuşma geçmişini ekle
    for entry in conversation_history:
        if 'role' in entry and 'content' in entry:
            messages.append({"role": entry['role'], "content": entry['content']})
    
    # Kullanıcının mevcut mesajını ekle
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=3000,
        temperature=0.7,
    )
    
    return response.choices[0].message.content

@app.route('/')
def index():
    """
    Ana sayfa - web arayüzünü gösterir.
    """
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def chat():
    """
    Genel sohbet için API endpoint'i.
    
    Beklenen JSON formatı:
    {
        "message": "Kullanıcı mesajı",
        "session_id": "Oturum ID'si (opsiyonel)",
        "conversation_history": "Önceki konuşma geçmişi (opsiyonel)",
        "model": "Model adı (opsiyonel, varsayılan: gpt-3.5-turbo)"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Geçersiz istek formatı. message alanı gerekli.'
            }), 400

        message = data['message']
        session_id = data.get('session_id', 'default')
        client_conversation_history = data.get('conversation_history', [])
        model_name = data.get('model', 'gpt-3.5-turbo')  # Varsayılan model
        
        # Konuşma geçmişini al
        server_conversation_history = get_conversation_history(session_id)
        
        # Konuşma geçmişini birleştir (client varsa onu öncelikle kullan)
        conversation_history = client_conversation_history if client_conversation_history else server_conversation_history
        
        # Basit sohbet yanıtı oluştur
        chat_response = simple_chat_response(message, conversation_history, model_name)
        
        # Konuşma geçmişini güncelle
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
            
        conversation_memory[session_id].append({
            'timestamp': datetime.now(),
            'role': 'user',
            'content': message
        })
        
        conversation_memory[session_id].append({
            'timestamp': datetime.now(),
            'role': 'assistant',
            'content': chat_response
        })
        
        return jsonify({
            'response': chat_response,
            'conversation_history': conversation_history,
            'model_used': model_name
        })
        
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_message = str(e)
        tb = traceback.format_exc()
        print(f"[HATA] {error_type}: {error_message}\n{tb}")
        return jsonify({
            'error': 'Bir hata oluştu.',
            'error_type': error_type,
            'error_message': error_message,
            'traceback': tb if os.environ.get('DEBUG') == '1' else None
        }), 500

@app.route('/api/drug-info', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def get_drug_info():
    """
    İlaç bilgisi ve soru için API endpoint'i.
    """
    try:
        data = request.get_json()
        
        if not data or 'drug_name_en' not in data or 'question_tr' not in data:
            return jsonify({
                'error': 'Geçersiz istek formatı. drug_name_en ve question_tr alanları gerekli.'
            }), 400

        drug_name_en = data['drug_name_en'].strip()
        question_tr = data['question_tr'].strip()
        session_id = data.get('session_id', 'default')
        client_conversation_history = data.get('conversation_history', [])
        model_name = data.get('model', 'gpt-3.5-turbo')  # Varsayılan model

        # İlaç adı kontrolü
        if not drug_name_en:
            return jsonify({
                'error': 'İlaç adı boş olamaz.',
                'web_response': 'Lütfen bir ilaç adı belirtin.'
            }), 400

        if len(drug_name_en.split()) > 3:
            return jsonify({
                'error': 'İlaç adı çok uzun.',
                'web_response': 'İlaç adı çok uzun görünüyor. Lütfen araştırmak istediğiniz ilacın adını daha kısa ve net belirtin.'
            }), 400

        # Konuşma geçmişini al ve birleştir
        server_conversation_history = get_conversation_history(session_id)
        conversation_history = client_conversation_history if client_conversation_history else server_conversation_history

        # API anahtarlarını al
        api_key_openai = get_openai_api_key()
        api_key_google, cse_id_google = get_google_api_keys()

        # Servisleri başlat
        ai_assistant = AIAssistant(api_key_openai)
        google_search = GoogleSearch(api_key_google, cse_id_google)

        # Yanıt verilerini hazırla
        response_data = {
            'drug_name': drug_name_en,
            'question': question_tr,
            'web_response': None,
            'conversation_history': conversation_history
        }

        # Web araması isteği var mı kontrol et
        should_search_web = any(keyword in question_tr.lower() for keyword in ['internet', 'araştır', 'web', 'online', 'bul', 'sitesi', 'site', 'kaynak', 'fda', 'google'])

        # Sadece web araması isteniyorsa Google araması yap
        if should_search_web:
            try:
                # İlaç adı ve soruyu birleştirerek ara
                search_queries = google_search.create_search_queries(question_tr, drug_name_en)
                all_results = []
                
                for query in search_queries:
                    results = google_search.search_web(query)
                    if results:
                        all_results.extend(results)
                        # Yeterli sonuç varsa döngüyü sonlandır
                        if len(all_results) >= 5:
                            break

                if all_results:
                    # Web yanıtını oluştur
                    answer_google = google_search.analyze_and_summarize(question_tr, all_results, drug_name_en)
                    if answer_google:
                        # Duygu analizi ve empati ekle
                        emotions, intensity = ai_assistant.detect_emotion_and_intensity(question_tr)
                        natural_response = generate_natural_response(drug_name_en, answer_google, conversation_history, question_tr, model_name)
                        response_data['web_response'] = natural_response
                        add_to_conversation_history(session_id, drug_name_en, question_tr, natural_response)
                    else:
                        response_data['web_response'] = (
                            f"Üzgünüm, {drug_name_en} hakkında bu soruya net bir cevap bulamadım. "
                            "Lütfen sorunuzu daha spesifik hale getirir misiniz?"
                        )
                else:
                    response_data['web_response'] = (
                        f"Üzgünüm, {drug_name_en} hakkında güvenilir bilgi bulamadım. "
                        "Lütfen ilaç adının doğru yazıldığından emin olun."
                    )
            except Exception as e:
                print(f"Web araması sırasında hata: {str(e)}")
                response_data['web_response'] = "Üzgünüm, şu anda bilgi alma konusunda teknik bir sorun yaşıyorum."

        # Hiçbir yanıt alınamadıysa
        if not response_data['web_response']:
            response_data['web_response'] = (
                f"Üzgünüm, {drug_name_en} hakkında şu anda bilgi edinemiyorum. "
                "Eğer ilaç hakkında bilgi almak istiyorsanız, lütfen 'araştır' veya 'internet' gibi kelimeler kullanarak sorunuzu tekrar sorun."
            )

        return jsonify({
            **response_data,
            'model_used': model_name
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        error_message = str(e)
        tb = traceback.format_exc()
        print(f"[HATA] {error_type}: {error_message}\n{tb}")
        return jsonify({
            'error': 'Bir hata oluştu.',
            'error_type': error_type,
            'error_message': error_message,
            'traceback': tb if os.environ.get('DEBUG') == '1' else None
        }), 500

@app.route('/ask', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def ask_question():
    """
    Spring Boot uygulaması için basit soru-cevap endpoint'i.
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Geçersiz istek formatı. question alanı gerekli.'
            }), 400

        question = data['question']
        model_name = data.get('model', 'gpt-3.5-turbo')  # Varsayılan model

        # Basit sohbet yanıtı oluştur
        chat_response = simple_chat_response(question, [], model_name)
        
        return jsonify({
            'answer': chat_response,
            'model_used': model_name
        })
        
    except Exception as e:
        return jsonify({'error': f"Bilinmeyen bir hata oluştu: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 