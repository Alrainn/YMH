## Devlogs - BIOWORKS Projesi

### Genel Açıklama

BIOWORKS projesi, kullanıcıların etkileşimde bulunabileceği bir web uygulamasıdır. Proje, kullanıcıların hesap oluşturmasını, giriş yapmasını, şifrelerini yönetmesini ve ana sayfada etkileşimde bulunmasını sağlar. Ayrıca, Hakkımızda ve İletişim sayfaları ile kullanıcıların bilgi alması ve iletişim kurması hedeflenmiştir.

**Önemli Not:** BIOWORKS projesi statik olarak yayınlanmıştır. 

### Dosya Yapısı

Proje dosyaları aşağıdaki gibi yapılandırılmıştır:

-   `index.html`: Giriş sayfası
-   `register.html`: Kayıt sayfası
-   `forgot_password.html`: Şifremi Unuttum sayfası
-   `main_page.html`: Ana sayfa
-   `about.html`: Hakkımızda sayfası
-   `communication.html`: İletişim sayfası
-   `css/`: Stil dosyaları
    -   `style.css`: Genel stiller
    -   `main_page.css`: Ana sayfaya özel stiller
    -   `profile.css`: Profil sayfasına özel stiller
    -   `communication.css`: İletişim sayfasına özel stiller
    -   `about.css`: Hakkımızda sayfasına özel stiller
-   `js/`: JavaScript dosyaları
    -   `bioscript.js`: Giriş ve kayıt sayfaları için şifre gösterme özelliği
    -   `communication.js`: İletişim sayfası için etkileşimler
-   `photos/`: Resim dosyaları
    -   `a-resim/`: Genel resimler ve logo

### Sayfa Detayları

#### 1. Giriş Sayfası (`index.html`)

-   Kullanıcıların e-posta ve şifre ile giriş yapmasını sağlar.
-   "Beni Hatırla" seçeneği bulunur.
-   "Şifremi Unuttum" sayfasına yönlendirme bağlantısı (`forgot_password.html`) içerir.
-   "Kayıt Ol" sayfasına yönlendirme bağlantısı (`register.html`) içerir.
-   "Misafir Olarak Devam Et" seçeneği (`main_page.html`) sunar.
-   Şifre alanında şifreyi gizleme/gösterme özelliği için JavaScript (`bioscript.js`) kullanılmıştır.
-   **Yeni:** Giriş formu alanlarına (`email`, `password`) ve butonlara daha modern bir görünüm kazandıran stiller eklendi.

#### 2. Kayıt Sayfası (`register.html`)

-   Kullanıcıların ad, soyad, yaş, cinsiyet, e-posta ve şifre bilgilerini girmesini sağlar.
-   Şifre doğrulama alanı bulunur.
-   "Giriş Yap" sayfasına yönlendirme bağlantısı (`index.html`) içerir.
-   Şifre alanında şifreyi gizleme/gösterme özelliği için JavaScript (`bioscript.js`) kullanılmıştır.
-   **Yeni:** Kayıt formu alanlarına (`name`, `surname`, `age`, `gender`, `email`, `password`, `confirm-password`) ve butonlara daha modern bir görünüm kazandıran stiller eklendi.

#### 3. Şifremi Unuttum Sayfası (`forgot_password.html`)

-   Kullanıcıların e-posta adreslerini girmesini sağlar.
-   Şifre sıfırlama bağlantısı gönderme butonu bulunur.
-   "Giriş Yap" (`index.html`) ve "Kayıt Ol" (`register.html`) sayfalarına yönlendirme bağlantıları içerir.
-   **Yeni:** E-posta giriş alanına ve butona modern bir görünüm kazandıran stiller eklendi.

#### 4. Ana Sayfa (`main_page.html`)

-   Sol panelde sohbet geçmişi bölümü bulunur.
-   Mesaj gönderme alanı içerir.
-   Üst menüde "Ana Sayfa" (`main_page.html`), "Hakkımızda" (`about.html`), "İletişim" (`communication.html`) ve "Profil" (`profile.html`) sayfalarına bağlantılar yer alır.
-   Font Awesome ikonları kullanılmıştır.
-   **Yeni:** Sayfa düzeni ve bileşenlere (sol panel, üst menü, sohbet alanı) modern bir görünüm kazandıran stiller eklendi.

#### 5. Hakkımızda Sayfası (`about.html`)

-   Uygulama ve geliştiricileri hakkında bilgiler içerir.
-   Ekip üyelerinin bilgileri ve sosyal medya bağlantıları yer alır.
-   **Yeni:** Hakkımızda sayfasına özel stiller (`about.css`) eklenerek takım üyeleri ve bilgi kutuları düzenlendi.

#### 6. İletişim Sayfası (`communication.html`)

-   Kullanıcıların iletişim kurabileceği bir form içerir.
-   Ad Soyad, E-posta, Konu ve Mesaj alanları bulunur.
-   Form gönderildiğinde mesaj gösterme özelliği için JavaScript (`communication.js`) kullanılmıştır.
-   **Yeni:** İletişim sayfasına özel stiller (`communication.css`) eklenerek iletişim formu ve bileşenler düzenlendi. Saat, motive edici sözler ve canvas animasyonu gibi etkileşimler için JavaScript (`communication.js`) güncellendi.

#### 7. Profil Sayfası (`profile.html`)

-   Kullanıcıların profil bilgilerini görüntüleyebileceği ve düzenleyebileceği bir arayüz sunar.
-   Profil fotoğrafını değiştirme, şifreyi güncelleme ve çıkış yapma seçenekleri bulunur.
-   **Yeni:** Profil sayfasına özel stiller (`profile.css`) eklenerek profil kartları, formlar ve butonlar düzenlendi.

### Stil Dosyaları

-   `style.css`: Tüm sayfaların genel görünümünü ve düzenini sağlamak için temel CSS stilleri içerir.
-   `main_page.css`: Ana sayfaya özel stiller içerir. Sol panel, üst panel, sohbet bölümü ve mesaj alanı gibi bileşenlerin stilleri tanımlanmıştır.
-   `profile.css`: Profil sayfasına özel stiller içerir. Profil kartları, formlar ve butonlar gibi öğelerin stilleri tanımlanmıştır.
-   `communication.css`: İletişim sayfasına özel stiller içerir. İletişim formu, etiketler ve butonlar gibi öğelerin stilleri tanımlanmıştır.
-   `about.css`: Hakkımızda sayfasına özel stiller içerir. Takım üyeleri ve bilgi kutuları gibi öğelerin stilleri tanımlanmıştır.
-   **Yeni:** Tüm stil dosyalarında renkler, yazı tipleri, boyutlar, hizalamalar ve görsel efektler güncellenerek modern bir tasarım elde edildi.

### JavaScript Dosyaları

-   `bioscript.js`: Giriş ve kayıt sayfalarındaki şifre alanlarında şifreyi gizleme/gösterme özelliğini sağlar.
-   `communication.js`: İletişim sayfasında saat, motive edici sözler ve canvas animasyonu gibi etkileşimleri yönetir.
-   **Yeni:** `communication.js` dosyasına iletişim sayfasında eklenen yeni etkileşimleri yönetmek için güncellemeler yapıldı.

### Genel Notlar

-   Projede Font Awesome kütüphanesi kullanılarak ikonlar eklenmiştir.
-   Tüm sayfaların temel düzeni ve stilleri `style.css` dosyası ile sağlanmıştır.
-   Her sayfa, kendi özel stil dosyası ile (örneğin, `main_page.css`, `profile.css`) ek olarak stilize edilmiştir.
-   **Yeni:** Projenin genelinde modern tasarım trendlerine uygun iyileştirmeler yapılmış ve kullanıcı deneyimi artırılmıştır.
-   **Yeni:** Site statik olarak yayınlanmıştır.

## 20 Mayıs 2025 - Ana Sayfa ve Sohbet Sistemi Geliştirmeleri

### 1. Dosya ve Kod Yapısının Modülerleştirilmesi
- Her sayfa için ayrı JS dosyası yapısına geçildi (`main_page.js`, `main_page_chatbox.js`).
- Sohbet geçmişi, sohbet kutusu ve diğer işlevler ayrıştırıldı.

### 2. Sohbet Geçmişi ve Chatbox Entegrasyonu
- Sohbet kutusuna tıklanınca o sohbete ait mesajların yüklenmesi sağlandı.
- Her sohbetin mesajları ayrı ayrı localStorage'da saklanacak şekilde düzenlendi.
- Aktif sohbet vurgulaması (yeşil çerçeve ve arka plan) eklendi.

### 3. Sohbet Arama Özelliği
- Sol panelde sohbet arama kutusu eklendi.
- Arama kutusuna yazdıkça başlık veya son mesaja göre sohbetler filtreleniyor.

### 4. Sohbet Sabitleme/Favori
- Üç nokta menüsüne "Sohbeti Sabitle / Sabit Kaldır" seçeneği eklendi.
- Sabitlenen sohbetler listenin en üstünde ve sarı çerçeve + yıldız ile vurgulanıyor.
- Menüde ikonlar (pushpin, ban, pen, trash) ile modern görünüm sağlandı.

### 5. Sohbet Silme ve İsmi Düzenleme
- Üç nokta menüsünde "İsmi Düzenle" ve "Sohbeti Sil" seçenekleri eklendi.
- Silme ve isim değiştirme işlemleri localStorage ve arayüzde anında güncelleniyor.

### 6. Tüm Sohbetleri Toplu Silme
- "Tüm Sohbetleri Sil" için modern bir çöp kutusu butonu eklendi.
- Buton, sohbet arama kutusunun sağında, küçük ve modern bir şekilde konumlandırıldı.
- Tıklanınca onay alınıyor ve tüm sohbetler ile mesajlar siliniyor.

### 7. Toast Bildirim Sistemi
- Sağ alt köşede, modern ve yuvarlatılmış bir toast/snackbar bildirimi eklendi.
- Sohbet silindiğinde, yeni sohbet oluşturulduğunda veya tüm sohbetler silindiğinde otomatik olarak bildirim çıkıyor.
- Başarı için yeşil, hata için kırmızı ikon ve arka plan kullanıldı.

### 8. Görsel ve UX İyileştirmeleri
- Sohbet kutuları, butonlar ve menülerde modern ve sade bir görünüm sağlandı.
- Responsive (mobil uyumlu) CSS düzenlemeleri yapıldı.
- Üç nokta menüsünde ikonlar ve renkli vurgular eklendi.
- Arama kutusu ve toplu silme butonu yan yana, uyumlu şekilde hizalandı.

### 9. Kod Temizliği ve Hata Giderme
- `activeChatId` değişkeninin global olarak paylaşılması sağlandı.
- Event listener çakışmaları ve tanım hataları giderildi.
- Sohbet kutusu ve sohbet geçmişi arasında tam entegrasyon sağlandı.

## 22 Mayıs 2024 - Profil ve Kullanıcı Deneyimi Geliştirmeleri

### 1. Profil Fotoğrafı Kırpma ve Kalıcı Saklama
- Kullanıcı profil fotoğrafı seçtiğinde Cropper.js ile kırpma ve önizleme özelliği eklendi.
- Kırpılan fotoğraf, localStorage'a base64 olarak kaydediliyor ve sayfa yenilense bile kaybolmuyor.
- Sol paneldeki küçük profil fotoğrafı da otomatik olarak güncelleniyor.
- Hakkımızda sayfasında da sol panelde aynı şekilde güncel profil fotoğrafı gösteriliyor.

### 2. Profil Bilgilerinin Kalıcı Saklanması
- Kullanıcı Ad Soyad ve E-posta bilgilerini güncellediğinde, bu bilgiler de localStorage'a kaydediliyor.
- Sayfa her yüklendiğinde localStorage'daki bilgiler otomatik olarak inputlara yükleniyor.
- Sol paneldeki "Hoş geldiniz, ..." kısmı da güncel kullanıcı adı ile otomatik değişiyor.
- Hakkımızda sayfasında da aynı şekilde güncel kullanıcı adı gösteriliyor.

### 3. Toast Bildirim Sistemi
- Profil ve şifre güncelleme işlemlerinde, ana sayfadaki gibi sağ altta modern toast/snackbar bildirimleri eklendi.
- Başarı için yeşil, hata için kırmızı arka plan ve animasyonlu geçişler kullanıldı.

### 4. Kullanıcı Deneyimi ve Kod İyileştirmeleri
- Tüm profil sayfası JS kodları, DOM yüklendikten sonra çalışacak şekilde düzenlendi (DOMContentLoaded).
- Script dosyaları ve modal yapısı, HTML'in en sonunda olacak şekilde taşındı.
- Kodlar sadeleştirildi ve tekrar eden işlemler fonksiyonel hale getirildi.

### 5. Tema Seçimi ve Profil Düzeni Özelliklerinin Kaldırılması
- Tema seçimi ve profil düzeni ile ilgili tüm butonlar, JS ve CSS kodları kaldırıldı.
- Arka plan görselleri ve gradyanlar kullanıldığı için tema seçimi gereksiz bulundu ve sadeleştirme yapıldı.