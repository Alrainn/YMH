package com.melihawci.springrestapi.repository;

import com.melihawci.springrestapi.model.ChatHistory;
import com.melihawci.springrestapi.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface ChatHistoryRepository extends JpaRepository<ChatHistory, Long> {
    List<ChatHistory> findByUserOrderByTimestampDesc(User user);
} 