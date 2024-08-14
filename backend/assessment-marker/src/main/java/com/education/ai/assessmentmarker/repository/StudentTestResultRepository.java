// StudentTestResultRepository.java: This repository interface extends 
// JpaRepository to handle CRUD operations for StudentTestResult entities.
// It includes custom methods to find results by student ID and test name.



package com.education.ai.assessmentmarker.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import com.education.ai.assessmentmarker.StudentTestResult;
import java.util.List;

public interface StudentTestResultRepository extends JpaRepository<StudentTestResult, Long> {
    List<StudentTestResult> findByStudentId(String studentId);
    List<StudentTestResult> findByStudentIdAndTestName(String studentId, String testName);
}
