package com.melihawci.springrestapi.service;

import com.melihawci.springrestapi.dto.ChatResponse;
import com.melihawci.springrestapi.model.ChatHistory;
import com.melihawci.springrestapi.model.User;
import com.melihawci.springrestapi.repository.ChatHistoryRepository;
import com.melihawci.springrestapi.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class ChatService {

    @Autowired
    private ChatHistoryRepository chatHistoryRepository;

    @Autowired
    private UserRepository userRepository;

    @Value("${ai.service.url}")
    private String aiServiceUrl;

    private final RestTemplate restTemplate = new RestTemplate();

    public ChatResponse processMessage(String message, Long userId) {
        try {
            // AI servisine istek gönder
            String aiResponse = restTemplate.postForObject(
                aiServiceUrl + "/chat",
                message,
                String.class
            );

            // Kullanıcıyı bul (misafir ise null)
            User user = userId != null ? userRepository.findById(userId).orElse(null) : null;

            // Sohbet geçmişini kaydet
            ChatHistory chatHistory = new ChatHistory();
            chatHistory.setUser(user);
            chatHistory.setUserMessage(message);
            chatHistory.setAiResponse(aiResponse);
            chatHistory.setTimestamp(LocalDateTime.now());
            chatHistoryRepository.save(chatHistory);

            // Yanıtı hazırla
            ChatResponse response = new ChatResponse();
            response.setId(chatHistory.getId());
            response.setUserMessage(message);
            response.setAiResponse(aiResponse);
            response.setTimestamp(chatHistory.getTimestamp());
            response.setStatus("success");

            return response;
        } catch (Exception e) {
            ChatResponse errorResponse = new ChatResponse();
            errorResponse.setStatus("error");
            errorResponse.setError("AI servisi ile iletişim kurulamadı: " + e.getMessage());
            return errorResponse;
        }
    }

    public List<ChatHistory> getUserChatHistory(Long userId) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("Kullanıcı bulunamadı"));
        return chatHistoryRepository.findByUserOrderByTimestampDesc(user);
    }
} 