// StudentAnswerRepository.java: This repository interface extends JpaRepository 
// to handle CRUD operations for StudentAnswer entities.
// It provides methods for saving, finding, and deleting student answers.



package com.education.ai.assessmentmarker.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.education.ai.assessmentmarker.StudentAnswer;

@Repository
public interface StudentAnswerRepository extends JpaRepository<StudentAnswer, Long> {}
