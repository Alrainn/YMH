package com.melihawci.springrestapi.controller;

import com.melihawci.springrestapi.dto.ChatRequest;
import com.melihawci.springrestapi.dto.ChatResponse;
import com.melihawci.springrestapi.model.ChatHistory;
import com.melihawci.springrestapi.service.ChatService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:3000")
public class ChatController {

    @Autowired
    private ChatService chatService;

    @PostMapping("/chat")
    public ResponseEntity<ChatResponse> chat(@RequestBody ChatRequest request) {
        try {
            ChatResponse response = chatService.processMessage(request.getMessage(), request.getUserId());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            ChatResponse errorResponse = new ChatResponse();
            errorResponse.setStatus("error");
            errorResponse.setError("Bir hata oluştu: " + e.getMessage());
            return ResponseEntity.internalServerError().body(errorResponse);
        }
    }

    @GetMapping("/history/{userId}")
    public ResponseEntity<?> getChatHistory(@PathVariable Long userId) {
        try {
            List<ChatHistory> history = chatService.getUserChatHistory(userId);
            List<ChatResponse> responses = history.stream()
                .map(chat -> {
                    ChatResponse response = new ChatResponse();
                    response.setId(chat.getId());
                    response.setUserMessage(chat.getUserMessage());
                    response.setAiResponse(chat.getAiResponse());
                    response.setTimestamp(chat.getTimestamp());
                    response.setStatus("success");
                    return response;
                })
                .collect(Collectors.toList());
            
            return ResponseEntity.ok(responses);
        } catch (RuntimeException e) {
            ChatResponse response = new ChatResponse();
            response.setStatus("error");
            response.setError(e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
} 