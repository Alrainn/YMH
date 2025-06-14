// Şifre göster/gizle fonksiyonu

document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.toggle-password').forEach(function(el) {
        el.addEventListener('click', function() {
            var input = document.querySelector(this.getAttribute('toggle'));
            if (input.type === 'password') {
                input.type = 'text';
                input.classList.add('password-visible');
                this.innerHTML = '<i class="fa fa-eye-slash"></i>';
            } else {
                input.type = 'password';
                input.classList.remove('password-visible');
                this.innerHTML = '<i class="fa fa-eye"></i>';
            }
        });
    });
}); 