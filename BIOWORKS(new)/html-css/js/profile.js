document.addEventListener('DOMContentLoaded', function() {
// Profil fotoğrafı değiştir butonuna tıklanınca dosya seçtir
if(document.getElementById('changePhotoBtn')) {
    document.getElementById('changePhotoBtn').onclick = function() {
        document.getElementById('profilePhotoInput').click();
    };
}

const profileImage = document.getElementById('profileImage');
const sideProfilePhoto = document.querySelector('.side-profile-photo');
const sideUsername = document.querySelector('.side-username');

// Sayfa yüklendiğinde localStorage'dan profil fotoğrafını yükle
if (profileImage) {
    const savedPhoto = localStorage.getItem('profilePhoto');
    if (savedPhoto) {
        profileImage.src = savedPhoto;
    }
}

// Sayfa yüklendiğinde sol paneldeki profil fotoğrafı ve kullanıcı adını güncelle
if (sideProfilePhoto) {
    const savedPhoto = localStorage.getItem('profilePhoto');
    if (savedPhoto) {
        sideProfilePhoto.src = savedPhoto;
    }
}
if (sideUsername) {
    const savedName = localStorage.getItem('profileFullName');
    if (savedName) {
        sideUsername.innerHTML = `Hoş geldiniz,<br><b>${savedName}</b>`;
    }
}

// Cropper.js ile profil fotoğrafı kırpma ve önizleme
let cropper = null;
const cropperModal = document.getElementById('cropperModal');
const cropperImage = document.getElementById('cropperImage');
const cropperCancelBtn = document.getElementById('cropperCancelBtn');
const cropperSaveBtn = document.getElementById('cropperSaveBtn');
const profilePhotoInput = document.getElementById('profilePhotoInput');

if (profilePhotoInput) {
    profilePhotoInput.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = function(ev) {
                cropperImage.src = ev.target.result;
                cropperModal.style.display = 'flex';
                if (cropper) cropper.destroy();
                cropper = new Cropper(cropperImage, {
                    aspectRatio: 1,
                    viewMode: 1,
                    background: false,
                    autoCropArea: 1,
                });
            }
            reader.readAsDataURL(e.target.files[0]);
        }
    });
}

if (cropperCancelBtn) {
    cropperCancelBtn.onclick = function() {
        if (cropper) cropper.destroy();
        cropperModal.style.display = 'none';
    };
}

if (cropperSaveBtn) {
    cropperSaveBtn.onclick = function() {
        if (cropper) {
            const canvas = cropper.getCroppedCanvas({ width: 300, height: 300 });
            const dataUrl = canvas.toDataURL('image/png');
            if (profileImage) profileImage.src = dataUrl;
            localStorage.setItem('profilePhoto', dataUrl);
            if (sideProfilePhoto) sideProfilePhoto.src = dataUrl;
            cropper.destroy();
            cropperModal.style.display = 'none';
        }
    };
}

// Tema değiştirme işlevi
const themeButtons = document.querySelectorAll('.theme-btn');
themeButtons.forEach(button => {
    button.addEventListener('click', function() {
        const theme = this.dataset.theme;
        document.body.setAttribute('data-theme', theme);
        // Aktif tema butonunu güncelle
        themeButtons.forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');
        // Tema tercihini localStorage'a kaydet
        localStorage.setItem('theme', theme);
    });
});

// Sayfa yüklendiğinde kaydedilmiş temayı uygula
const savedTheme = localStorage.getItem('theme');
if (savedTheme) {
    document.body.setAttribute('data-theme', savedTheme);
    const activeThemeBtn = document.querySelector(`.theme-btn[data-theme="${savedTheme}"]`);
    if(activeThemeBtn) activeThemeBtn.classList.add('active');
}

// Profil düzeni değiştirme işlevi
const layoutButtons = document.querySelectorAll('.layout-btn');
layoutButtons.forEach(button => {
    button.addEventListener('click', function() {
        const layout = this.dataset.layout;
        // Aktif düzen butonunu güncelle
        layoutButtons.forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');
        // Düzen tercihini localStorage'a kaydet
        localStorage.setItem('layout', layout);
        // Düzen değişikliğini uygula
        document.querySelector('.profile-blocks-area').className = 
            `profile-blocks-area layout-${layout}`;
    });
});

// Sayfa yüklendiğinde kaydedilmiş düzeni uygula
const savedLayout = localStorage.getItem('layout');
if (savedLayout) {
    const activeLayoutBtn = document.querySelector(`.layout-btn[data-layout="${savedLayout}"]`);
    if(activeLayoutBtn) activeLayoutBtn.classList.add('active');
    const blocksArea = document.querySelector('.profile-blocks-area');
    if(blocksArea) blocksArea.className = `profile-blocks-area layout-${savedLayout}`;
}

// Toast/snackbar fonksiyonu
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast-notification');
    if (!toast) return;
    toast.textContent = message;
    toast.className = 'toast-show ' + (type === 'error' ? 'toast-error' : 'toast-success');
    setTimeout(() => {
        toast.className = '';
        toast.textContent = '';
    }, 2500);
}

// Profil bilgileri güncelleme butonu işlevi
const fullNameInput = document.getElementById('fullName');
const emailInput = document.getElementById('email');
const updateBtn = document.querySelector('.profile-form-wide .save-btn');

// Sayfa yüklendiğinde localStorage'dan profil bilgilerini yükle
if (fullNameInput && emailInput) {
    const savedName = localStorage.getItem('profileFullName');
    const savedEmail = localStorage.getItem('profileEmail');
    if (savedName) fullNameInput.value = savedName;
    if (savedEmail) emailInput.value = savedEmail;
}

if(fullNameInput && emailInput && updateBtn) {
    // Başlangıçta inputlar pasif
    fullNameInput.disabled = true;
    emailInput.disabled = true;
    let editing = false;

    updateBtn.addEventListener('click', function(e) {
        e.preventDefault();
        if(!editing) {
            // Düzenleme moduna geç
            fullNameInput.disabled = false;
            emailInput.disabled = false;
            fullNameInput.focus();
            updateBtn.innerHTML = '<i class="fa-solid fa-check"></i> Onayla';
            editing = true;
        } else {
            // Hata kontrolü
            const emailVal = emailInput.value.trim();
            const nameVal = fullNameInput.value.trim();
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!nameVal) {
                showToast('Ad Soyad boş olamaz!', 'error');
                fullNameInput.focus();
                return;
            }
            if (!emailRegex.test(emailVal)) {
                showToast('Geçerli bir e-posta girin!', 'error');
                emailInput.focus();
                return;
            }
            // Onayla: tekrar pasif yap
            fullNameInput.disabled = true;
            emailInput.disabled = true;
            updateBtn.innerHTML = '<i class="fa-solid fa-save"></i> Bilgileri Güncelle';
            editing = false;
            // Bilgileri localStorage'a kaydet
            localStorage.setItem('profileFullName', nameVal);
            localStorage.setItem('profileEmail', emailVal);
            if (sideUsername) {
                sideUsername.innerHTML = `Hoş geldiniz,<br><b>${nameVal}</b>`;
            }
            showToast('Bilgiler başarıyla güncellendi!');
        }
    });
}

// Şifre değiştirme formu için pasif/aktif ve buton ismi işlemleri
const passwordForm = document.querySelector('.password-form');
if(passwordForm) {
    const currentPassword = document.getElementById('currentPassword');
    const newPassword = document.getElementById('newPassword');
    const confirmPassword = document.getElementById('confirmPassword');
    const passwordBtn = passwordForm.querySelector('.save-btn');
    let editingPassword = false;
    if(currentPassword && newPassword && confirmPassword && passwordBtn) {
        // Başlangıçta inputlar pasif
        currentPassword.disabled = true;
        newPassword.disabled = true;
        confirmPassword.disabled = true;
        passwordBtn.addEventListener('click', function(e) {
            e.preventDefault();
            if(!editingPassword) {
                // Aktif et
                currentPassword.disabled = false;
                newPassword.disabled = false;
                confirmPassword.disabled = false;
                currentPassword.focus();
                passwordBtn.innerHTML = '<i class="fa-solid fa-check"></i> Onayla';
                editingPassword = true;
            } else {
                // Hata kontrolü
                if (!currentPassword.value.trim() || !newPassword.value.trim() || !confirmPassword.value.trim()) {
                    showToast('Tüm şifre alanlarını doldurun!', 'error');
                    return;
                }
                if (newPassword.value !== confirmPassword.value) {
                    showToast('Yeni şifreler uyuşmuyor!', 'error');
                    return;
                }
                if (newPassword.value.length < 6) {
                    showToast('Şifre en az 6 karakter olmalı!', 'error');
                    return;
                }
                // Tekrar pasif yap
                currentPassword.disabled = true;
                newPassword.disabled = true;
                confirmPassword.disabled = true;
                passwordBtn.innerHTML = '<i class="fa-solid fa-key"></i> Şifreyi Güncelle';
                editingPassword = false;
                showToast('Şifre başarıyla güncellendi!');
            }
        });
    }
}
}); 