// Ana sayfa için temel işlevsellik
document.addEventListener('DOMContentLoaded', function() {
    // Sohbet geçmişi yönetimi
    initializeChatHistory();
    // Yeni sohbet başlatma
    initializeNewChat();
    // Sohbet silme/düzenleme
    initializeChatActions();

    // Sohbet arama işlevselliği
    const searchInput = document.getElementById('chatSearchInput');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.toLowerCase();
            let chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
            if (query) {
                chatHistory = chatHistory.filter(chat =>
                    (chat.title && chat.title.toLowerCase().includes(query)) ||
                    (chat.lastMessage && chat.lastMessage.toLowerCase().includes(query))
                );
            }
            renderChatHistory(chatHistory);
        });
    }

    // Tüm sohbetleri sil butonu işlevi
    const deleteAllBtn = document.querySelector('.delete-all-chats-btn');
    if (deleteAllBtn) {
        deleteAllBtn.addEventListener('click', function() {
            if (confirm('Tüm sohbetleri ve mesajları silmek istediğinize emin misiniz? Bu işlem geri alınamaz!')) {
                localStorage.removeItem('chatHistoryList');
                Object.keys(localStorage).forEach(key => {
                    if (key.startsWith('chatMessages_')) {
                        localStorage.removeItem(key);
                    }
                });
                renderChatHistory([]);
                const chatMessages = document.getElementById('chatMessages');
                if (chatMessages) chatMessages.innerHTML = '';
                if (typeof activeChatId !== 'undefined') activeChatId = null;
                showToast('Tüm sohbetler silindi.', 'success');
            }
        });
    }

    // Yeni sohbet ekleme toast
    const newChatBtn = document.querySelector('.new-chat-button');
    if (newChatBtn) {
        newChatBtn.addEventListener('click', function() {
            setTimeout(() => {
                showToast('Yeni sohbet oluşturuldu.', 'success');
            }, 100);
        });
    }
});

// Sohbet geçmişini yükleme
function initializeChatHistory() {
    const chatList = document.querySelector('.chat-list');
    // Burada localStorage veya API'den sohbet geçmişini yükleyeceğiz
    loadChatHistory();
}

// Yeni sohbet başlatma
function initializeNewChat() {
    const newChatBtn = document.querySelector('.new-chat-button');
    newChatBtn.addEventListener('click', function() {
        // 1. Yeni sohbet objesi oluştur
        const chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
        const newChat = {
            id: Date.now(),
            title: `Sohbet ${chatHistory.length + 1}`,
            lastMessage: '',
            timestamp: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
        };
        chatHistory.unshift(newChat); // En üste ekle
        localStorage.setItem('chatHistoryList', JSON.stringify(chatHistory));

        // 2. Sohbet listesini güncelle
        renderChatHistory(chatHistory);

        // 3. Sohbet kutusunu sıfırla (mesajları temizle)
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) chatMessages.innerHTML = '';

        // 4. (İsteğe bağlı) Aktif sohbeti işaretle veya başka bir işlem yap
    });
}

// Sohbet silme/düzenleme işlevleri
function initializeChatActions() {
    // Gerekirse ek işlevler eklenebilir
}

// Sohbet geçmişini yükleme
function loadChatHistory() {
    const chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
    renderChatHistory(chatHistory);
}

let activeChatId = null;

function renderChatHistory(history) {
    // Sabitlenen sohbetler en üste gelsin
    history = [...history].sort((a, b) => (b.pinned === true) - (a.pinned === true));
    const chatList = document.querySelector('.chat-list');
    chatList.innerHTML = '';
    history.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        if (chat.id === activeChatId) {
            chatItem.classList.add('active-chat-item');
        }
        if (chat.pinned) {
            chatItem.classList.add('pinned-chat-item');
        }
        chatItem.innerHTML = `
            <div class=\"chat-preview\" style=\"display: flex; justify-content: space-between; align-items: flex-start;\">
                <div>
                    <h3>${chat.title}</h3>
                    <p>${chat.lastMessage}</p>
                </div>
                <button class=\"chat-menu-btn\" data-id=\"${chat.id}\" title=\"Menü\" style=\"background: none; border: none; font-size: 1.3em; color: #fff; cursor: pointer; margin-left: 10px;\">&#8942;</button>
            </div>
            <span class=\"chat-time\">${chat.timestamp}</span>
        `;
        chatItem.addEventListener('click', function(e) {
            if (!e.target.classList.contains('chat-menu-btn')) {
                selectChat(chat.id);
            }
        });
        chatList.appendChild(chatItem);
    });
    document.querySelectorAll('.chat-menu-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const id = Number(this.getAttribute('data-id'));
            showChatMenu(this, id);
        });
    });
}

function selectChat(chatId) {
    activeChatId = chatId;
    // Aktif sohbeti vurgula
    const chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
    renderChatHistory(chatHistory);
    // Mesajları yükle
    loadChatMessages(chatId);
}

function loadChatMessages(chatId) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    const messages = JSON.parse(localStorage.getItem('chatMessages_' + chatId) || '[]');
    chatMessages.innerHTML = '';
    messages.forEach(msg => {
        const rowDiv = document.createElement('div');
        rowDiv.className = msg.isUser ? 'message-row user-message' : 'message-row system-message';
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';
        bubbleDiv.innerHTML = msg.message.replace(/\n/g, '<br>');
        rowDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(rowDiv);
    });
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showChatMenu(button, chatId) {
    document.querySelectorAll('.chat-popup-menu').forEach(el => el.remove());
    let chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
    const chat = chatHistory.find(c => c.id === chatId);
    const isPinned = chat && chat.pinned;
    const menu = document.createElement('div');
    menu.className = 'chat-popup-menu';
    menu.style.position = 'absolute';
    menu.style.top = (button.getBoundingClientRect().top + window.scrollY + 30) + 'px';
    menu.style.left = (button.getBoundingClientRect().left + window.scrollX - 60) + 'px';
    menu.style.background = '#222';
    menu.style.color = '#fff';
    menu.style.borderRadius = '8px';
    menu.style.boxShadow = '0 2px 8px rgba(0,0,0,0.18)';
    menu.style.padding = '8px 0';
    menu.style.zIndex = 9999;
    menu.style.minWidth = '140px';
    menu.innerHTML = `
        <div class=\"chat-popup-menu-item\" style=\"padding: 8px 18px; cursor: pointer; display: flex; align-items: center;\">
            <i class=\"fa-solid ${isPinned ? 'fa-ban' : 'fa-thumbtack'}\" style=\"margin-right:8px;\"></i>${isPinned ? 'Sabit Kaldır' : 'Sohbeti Sabitle'}
        </div>
        <div class=\"chat-popup-menu-item\" style=\"padding: 8px 18px; cursor: pointer; display: flex; align-items: center;\">
            <i class=\"fa fa-pen\" style=\"margin-right:8px;\"></i>İsmi Düzenle
        </div>
        <div class=\"chat-popup-menu-item\" style=\"padding: 8px 18px; cursor: pointer; color: #ff6b6b; display: flex; align-items: center;\">
            <i class=\"fa fa-trash\" style=\"margin-right:8px;\"></i>Sohbeti Sil
        </div>
    `;
    document.body.appendChild(menu);
    const items = menu.querySelectorAll('.chat-popup-menu-item');
    // Sabitleme
    items[0].addEventListener('click', function() {
        togglePinChat(chatId);
        menu.remove();
    });
    // İsmi düzenle
    items[1].addEventListener('click', function() {
        menu.remove();
        showRenamePrompt(chatId);
    });
    // Silme
    items[2].addEventListener('click', function() {
        deleteChatById(chatId);
        menu.remove();
    });
    setTimeout(() => {
        document.addEventListener('click', function closeMenu(e) {
            if (!menu.contains(e.target) && e.target !== button) {
                menu.remove();
                document.removeEventListener('click', closeMenu);
            }
        });
    }, 10);
}

function showRenamePrompt(chatId) {
    const newName = prompt('Yeni sohbet ismini girin:');
    if (newName && newName.trim().length > 0) {
        let chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
        chatHistory = chatHistory.map(chat => {
            if (chat.id === chatId) {
                return {...chat, title: newName};
            }
            return chat;
        });
        localStorage.setItem('chatHistoryList', JSON.stringify(chatHistory));
        renderChatHistory(chatHistory);
    }
}

function deleteChatById(id) {
    let chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
    chatHistory = chatHistory.filter(chat => chat.id !== id);
    localStorage.setItem('chatHistoryList', JSON.stringify(chatHistory));
    renderChatHistory(chatHistory);
    showToast('Sohbet silindi.', 'success');
}

function togglePinChat(chatId) {
    let chatHistory = JSON.parse(localStorage.getItem('chatHistoryList') || '[]');
    chatHistory = chatHistory.map(chat => {
        if (chat.id === chatId) {
            return { ...chat, pinned: !chat.pinned };
        }
        return chat;
    });
    localStorage.setItem('chatHistoryList', JSON.stringify(chatHistory));
    renderChatHistory(chatHistory);
}

function showToast(message, type = 'success') {
    const toast = document.getElementById('toast-notification');
    if (!toast) return;
    toast.innerHTML = '';
    let icon = '';
    if (type === 'success') icon = '<span class="toast-icon"><i class="fa fa-check-circle"></i></span>';
    if (type === 'error') icon = '<span class="toast-icon"><i class="fa fa-times-circle"></i></span>';
    toast.className = '';
    toast.classList.add('toast-show', `toast-${type}`);
    toast.innerHTML = icon + message;
    setTimeout(() => {
        toast.classList.remove('toast-show');
    }, 2500);
} 