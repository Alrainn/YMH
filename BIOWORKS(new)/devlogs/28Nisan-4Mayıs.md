# BIOWORKS - Geliştirme Günlüğü - 2024-05-04 (28 Nisan - 4 Mayıs)

Bu hafta (28 Nisan - 4 Mayıs) BIOWORKS projesinde yapılan geliştirmeler:

## Genel İlerleme

Bu hafta, BIOWORKS projesinin temel HTML ve CSS yapısı üzerine odaklanıldı. Kullanıcıların etkileşimde bulunabileceği çeşitli sayfalar oluşturuldu.

## Tamamlanan İşler

* **Giriş Sayfası (`index.html`):**

    * Kullanıcıların e-posta ve şifrelerini girerek sisteme giriş yapmalarını sağlayan temel arayüz oluşturuldu.
    * "Beni Hatırla" seçeneği eklendi.
    * "Şifremi Unuttum" bağlantısı eklendi.
    * "Kayıt Ol" sayfasına yönlendirme bağlantısı eklendi.
    * "Misafir Olarak Devam Et" seçeneği eklendi.
    * Giriş formu düzenlendi ve stilize edildi.
    * "Şifremi Unuttum" bağlantısı için `forgot_password.html` sayfasına yönlendirme yapıldı.

* **Kayıt Sayfası (`register.html`):**

    * Kullanıcıların ad, soyad, yaş, cinsiyet, e-posta ve şifre bilgilerini girebilecekleri kayıt formu oluşturuldu.
    * Şifre doğrulama alanı eklendi.
    * "Giriş Yap" sayfasına yönlendirme bağlantısı eklendi.
    * Kayıt formu düzenlendi ve stilize edildi.

* **Şifremi Unuttum Sayfası (`forgot_password.html`):**

    * Kullanıcıların şifre sıfırlama talebinde bulunabilecekleri sayfa oluşturuldu.
    * E-posta adresi giriş alanı ve şifre sıfırlama bağlantısı gönderme butonu eklendi.
    * "Giriş Yap" ve "Kayıt Ol" sayfalarına yönlendirme bağlantıları eklendi.
    * Şifremi Unuttum sayfası düzenlendi ve stilize edildi.

* **Ana Sayfa (`main_page.html`):**

    * Sol tarafta sohbet geçmişinin görüntüleneceği temel arayüz oluşturuldu.
    * Üst menüde "Ana Sayfa", "Hakkımızda", "İletişim" ve "Profil" sayfalarına bağlantılar eklendi.
    * Mesaj gönderme alanı oluşturuldu.
    * Ana sayfa düzeni ve stilleri (`main_page.css`) iyileştirildi.
    * Font Awesome ikonları entegre edildi.

* **Profil Sayfası (`profile.html`):**

    * Kullanıcıların profil bilgilerini görüntüleyebileceği ve düzenleyebileceği temel sayfa oluşturuldu.
    * Profil fotoğrafı değiştirme, şifre güncelleme ve çıkış yapma bölümleri eklendi.
    * Profil sayfası stilize edildi (`profile.css`).
    * Soldaki bara kullanıcı dostu eklemeler yapıldı. (Küçük profil , küçük navigasyon tuşları)

* **Hakkımızda Sayfası (`about_us.html`):**

    * Uygulama hakkında bilgilerin yer alacağı temel sayfa oluşturuldu (içerik henüz eklenmedi).

* **İletişim Sayfası (`communication.html`):**

    * Kullanıcıların iletişim bilgilerine ulaşabileceği ve mesaj gönderebileceği temel sayfa oluşturuldu.
    * İletişim formu eklendi.
    * İletişim sayfası stilize edildi (`communication.css`).
    * Soldaki bara görsel açıdan iyileştirmeler yapıldı. 

* **Stil Dosyaları (`style.css`, `main_page.css`, `profile.css`, `communication.css`):**

    * Tüm sayfaların genel görünümünü ve düzenini sağlamak için temel CSS stilleri oluşturuldu (`style.css`).
    * Ana sayfaya özel stiller eklendi (`main_page.css`).
    * Profil sayfasına özel stiller eklendi (`profile.css`).
    * İletişim sayfasına özel stiller eklendi (`communication.css`).
    * Font Awesome kütüphanesi entegre edildi (ikonlar için).
    * Genel stil iyileştirmeleri ve düzenlemeler yapıldı.
    * `communication.css` dosyası içinde iletişim sayfası başlık bloğu, form etiketleri, giriş alanları, gönder butonu ve form mesajları için stiller tanımlandı.
    * `profile.css` dosyası içinde profil sayfası düzeni, sol panel, ana içerik alanı, navigasyon, profil kartları, formlar ve butonlar için stiller tanımlandı.
    * Tüm stil dosyalarında renkler, yazı tipleri, boyutlar, hizalamalar ve görsel efektler düzenlendi.

Bu güncellemeler, mevcut kodunuzdaki değişiklikleri ve eklemeleri yansıtacak şekilde yapıldı. Umarım yardımcı olur!

## Karşılaşılan Zorluklar

* CSS düzenlemelerinde bazı uyumluluk sorunları yaşandı, ancak çözüldü.
* Ana sayfa düzeninin istenen şeffaflık ve bulanıklık efektlerine sahip olması biraz zaman aldı.

## Gelecek Hafta Planları

* Veritabanı entegrasyonuna başlanacak.
* Kullanıcı etkileşimlerini yönetecek JavaScript kodları yazılacak.
* Duyarlılık (responsive) tasarım iyileştirmeleri yapılacak.

## Ek Notlar

Bu hafta projenin iskeleti tamamlandı. Gelecek hafta daha çok işlevsellik eklemeye odaklanacağız.

## Katkıda Bulunanlar

* [Salih Eren Çavuşoğlu , Berkay Hasret]