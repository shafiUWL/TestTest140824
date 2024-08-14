// StudentAnswerController.java: This controller class handles HTTP requests 
// related to student answers.
// It autowires the StudentAnswerService to save student answers.
// The endpoint "/submit-answer" accepts POST requests with a StudentAnswer 
// body and saves it to the repository.


package com.education.ai.assessmentmarker.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.education.ai.assessmentmarker.StudentAnswer;
import com.education.ai.assessmentmarker.service.StudentAnswerService;

@RestController
@RequestMapping("/api")
public class StudentAnswerController {
    @Autowired
    private StudentAnswerService studentAnswerService;

    @PostMapping("/submit-answer")
    public ResponseEntity<String> submitAnswer(@RequestBody StudentAnswer answer) {
        studentAnswerService.saveAnswer(answer);
        return ResponseEntity.ok("Answer recorded");
    }
}
