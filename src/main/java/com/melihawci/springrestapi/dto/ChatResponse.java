package com.melihawci.springrestapi.dto;

import java.time.LocalDateTime;

public class ChatResponse {
    private Long id;
    private String userMessage;
    private String aiResponse;
    private LocalDateTime timestamp;
    private String status;
    private String error;

    public Long getId() {
        return id;
    }
    public void setId(Long id) {
        this.id = id;
    }
    public String getUserMessage() {
        return userMessage;
    }
    public void setUserMessage(String userMessage) {
        this.userMessage = userMessage;
    }
    public String getAiResponse() {
        return aiResponse;
    }
    public void setAiResponse(String aiResponse) {
        this.aiResponse = aiResponse;
    }
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    public void setTimestamp(LocalDateTime timestamp) {
        this.timestamp = timestamp;
    }
    public String getStatus() {
        return status;
    }
    public void setStatus(String status) {
        this.status = status;
    }
    public String getError() {
        return error;
    }
    public void setError(String error) {
        this.error = error;
    }
} 