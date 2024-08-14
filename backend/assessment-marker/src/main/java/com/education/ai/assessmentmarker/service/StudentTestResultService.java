package com.education.ai.assessmentmarker.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.education.ai.assessmentmarker.StudentTestResult;
import com.education.ai.assessmentmarker.repository.StudentTestResultRepository;
import java.util.List;

@Service
public class StudentTestResultService {

    @Autowired
    private StudentTestResultRepository repository;

    public StudentTestResult saveResult(StudentTestResult result) {
        List<StudentTestResult> existingResults = repository.findByStudentIdAndTestName(result.getStudentId(), result.getTestName());
        result.setAttempts(existingResults.size() + 1);
        return repository.save(result);
    }

    public List<StudentTestResult> getResultsByStudentId(String studentId) {
        return repository.findByStudentId(studentId);
    }
}
