package com.education.ai.assessmentmarker.service;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.education.ai.assessmentmarker.StudentAnswer;
import com.education.ai.assessmentmarker.repository.StudentAnswerRepository;

@Service
public class StudentAnswerService {
    @Autowired
    private StudentAnswerRepository studentAnswerRepository;

    public void saveAnswer(StudentAnswer answer) {
        studentAnswerRepository.save(answer);
    }
}
