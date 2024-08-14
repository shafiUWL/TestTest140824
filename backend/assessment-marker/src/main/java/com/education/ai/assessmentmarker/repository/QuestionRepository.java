// QuestionRepository.java: This repository interface extends JpaRepository 
// to handle CRUD operations for Question entities.
// It provides methods for saving, finding, and deleting questions.



package com.education.ai.assessmentmarker.repository;

import com.education.ai.assessmentmarker.model.Question;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface QuestionRepository extends JpaRepository<Question, Long> {
}
