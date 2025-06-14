package com.melihawci.springrestapi.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;

@Service
public class AIService {
    
    @Value("${ai.service.url}")
    private String aiServiceUrl;
    
    private final RestTemplate restTemplate;
    
    public AIService() {
        this.restTemplate = new RestTemplate();
    }
    
    public String getAIResponse(String message) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            Map<String, String> requestBody = new HashMap<>();
            requestBody.put("message", message);

            HttpEntity<Map<String, String>> request = new HttpEntity<>(requestBody, headers);

            String response = restTemplate.postForObject(
                aiServiceUrl + "/chat",
                request,
                String.class
            );
            return response != null ? response : "Üzgünüm, şu anda yanıt veremiyorum.";
        } catch (Exception e) {
            e.printStackTrace();
            return "Yapay zeka servisi ile iletişim kurulurken bir hata oluştu: " + e.getMessage();
        }
    }
} 