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
import logging

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        logger.error("OpenAI API anahtarı bulunamadı")
        raise ValueError("OpenAI API anahtarı bulunamadı")
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

def generate_natural_response(drug_name: str, raw_response: str, conversation_history: List[Dict], question: str) -> str:
    """
    FDA veya Google'dan gelen ham yanıtı alıp daha doğal, sohbet tarzında bir yanıt oluşturur.
    """
    api_key: str = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    
    # Konuşma geçmişini OpenAI formatına dönüştür
    messages = [
        {"role": "system", "content": f"""
        Sen ilaçlar hakkında bilgi veren, samimi ve yardımsever bir sağlık asistanısın.
        Kullanıcının sorduğu sorulara aşağıdaki bilgilere dayanarak cevap vereceksin.
        Cevaplarında resmi tıbbi dil yerine, daha samimi ve anlaşılır bir dil kullan.
        İlaç bilgilerini sunarken fazla teknik terimleri açıkla ve kullanıcının endişelerini dinle.
        Bilgi kaynağının FDA veya web aramaları olduğunu belirtmene gerek yok.
        Cevaplarında kısa ve öz ol, ancak önemli bilgileri atlamadan ve samimi bir dilde yanıtla.
        
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
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1500,
        temperature=0.7,
    )
    
    return response.choices[0].message.content

def simple_chat_response(message: str, conversation_history: List[Dict]) -> str:
    """
    İlaç araması yapmadan basit sohbet yanıtları üretir.
    """
    api_key: str = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    
    messages = [
        {"role": "system", "content": """
        Sen samimi ve yardımsever bir sağlık asistanısın. Sağlık ve ilaçlar konusunda genel bilgiler verebilirsin,
        ancak spesifik ilaç tavsiyeleri yapmamalısın. Kullanıcıyla doğal bir sohbet akışı içinde
        konuşmalısın. Eğer kullanıcı belirli bir ilaç hakkında bilgi isterse, ona yardımcı olmak istediğini
        ama önce ilaç adını açıkça belirtmesi gerektiğini söyle.
        
        Yanıtlarını kısa ve anlaşılır tut, tıbbi terimleri kullanmaktan kaçın. Kullanıcıya
        bir doktor veya eczacıya danışması gerektiğini hatırlatabilirsin.
        """}
    ]
    
    # Konuşma geçmişini ekle
    for entry in conversation_history:
        if 'role' in entry and 'content' in entry:
            messages.append({"role": entry['role'], "content": entry['content']})
    
    # Kullanıcının mevcut mesajını ekle
    messages.append({"role": "user", "content": message})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    
    return response.choices[0].message.content

@app.route('/')
def index():
    """
    Ana sayfa - web arayüzünü gösterir.
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Geçersiz istek formatı")
            return jsonify({
                'error': 'Geçersiz istek formatı'
            }), 400

        message = data['message'].strip()
        if not message:
            logger.warning("Boş mesaj gönderildi")
            return jsonify({
                'error': 'Mesaj boş olamaz'
            }), 400

        api_key = get_openai_api_key()
        client = OpenAI(api_key=api_key)

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    Sen samimi ve yardımsever bir sağlık asistanısın. Sağlık ve ilaçlar konusunda genel bilgiler verebilirsin,
                    ancak spesifik ilaç tavsiyeleri yapmamalısın. Kullanıcıyla doğal bir sohbet akışı içinde konuşmalısın.
                    Yanıtlarını kısa ve anlaşılır tut, tıbbi terimleri kullanmaktan kaçın. Kullanıcıya bir doktor veya
                    eczacıya danışması gerektiğini hatırlatabilirsin.
                    """},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7,
            )

            ai_response = response.choices[0].message.content
            logger.info("AI yanıtı başarıyla oluşturuldu")
            return ai_response

        except Exception as e:
            logger.error(f"OpenAI API hatası: {str(e)}")
            return jsonify({
                'error': f'AI yanıtı oluşturulamadı: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Genel hata: {str(e)}")
        return jsonify({
            'error': f'Bir hata oluştu: {str(e)}'
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
            'fda_response': None,
            'web_response': None,
            'conversation_history': conversation_history
        }

        # FDA bilgilerini al
        drug = Drug(drug_name_en)
        fda_success = drug.get_fda_info()
        
        if fda_success:
            try:
                drug.clean_data()
                drug_info = {
                    "substance_name": drug.substance_name,
                    "indications_and_usage": drug.indications_and_usage,
                    "warnings": drug.warnings,
                    "dosage_and_administration": drug.dosage_and_administration,
                    "adverse_reactions": drug.adverse_reactions
                }

                # FDA yanıtını oluştur
                prompt = AIAssistant.create_prompt(drug_info, question_tr)
                response_fda = ai_assistant.get_response(prompt)
                
                if response_fda:
                    # Duygu analizi ve empati ekle
                    emotions, intensity = ai_assistant.detect_emotion_and_intensity(question_tr)
                    natural_response = ai_assistant.create_empathetic_response(response_fda, emotions, intensity)
                    response_data['fda_response'] = natural_response
                    add_to_conversation_history(session_id, drug_name_en, question_tr, natural_response)
                else:
                    print("FDA yanıtı alınamadı, web aramasına geçiliyor...")
            except Exception as e:
                print(f"FDA yanıtı işlenirken hata: {str(e)}")

        # Web araması yap (FDA yanıtı yoksa veya yetersizse)
        if not response_data['fda_response']:
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
                        natural_response = ai_assistant.create_empathetic_response(answer_google, emotions, intensity)
                        response_data['web_response'] = natural_response
                        add_to_conversation_history(session_id, drug_name_en, question_tr, natural_response)
                    else:
                        response_data['web_response'] = (
                            f"Üzgünüm, {drug_name_en} hakkında bu soruya net bir cevap bulamadım. "
                            "Lütfen doktorunuza veya eczacınıza danışın."
                        )
                else:
                    response_data['web_response'] = (
                        f"Üzgünüm, {drug_name_en} hakkında güvenilir bilgi bulamadım. "
                        "Lütfen ilaç adının doğru yazıldığından emin olun veya bir sağlık profesyoneline danışın."
                    )
            except Exception as e:
                print(f"Web araması sırasında hata: {str(e)}")
                response_data['web_response'] = "Üzgünüm, şu anda bilgi alma konusunda teknik bir sorun yaşıyorum."

        # Hiçbir yanıt alınamadıysa
        if not response_data['fda_response'] and not response_data['web_response']:
            response_data['web_response'] = (
                f"Üzgünüm, {drug_name_en} hakkında şu anda bilgi edinemiyorum. "
                "Lütfen daha sonra tekrar deneyin veya bir sağlık profesyoneline danışın."
            )

        return jsonify(response_data)

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
        question = request.get_data(as_text=True)
        if not question:
            return jsonify({
                'error': 'Soru boş olamaz.'
            }), 400

        # Basit sohbet yanıtı oluştur
        chat_response = simple_chat_response(question, [])
        
        return jsonify({
            'answer': chat_response
        })
        
    except Exception as e:
        return jsonify({'error': f"Bilinmeyen bir hata oluştu: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 