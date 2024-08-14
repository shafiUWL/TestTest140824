// QuestionController.java: This controller class manages questions.
// It provides endpoints to submit questions via POST and retrieve all 
// questions via GET.
// The endpoint "/submit-questions" accepts a list of Question objects 
// and saves them to the repository.
// The endpoint "/questions" returns a list of all questions stored in 
// the repository.


package com.education.ai.assessmentmarker.controller;

import com.education.ai.assessmentmarker.model.Question;
import com.education.ai.assessmentmarker.repository.QuestionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@CrossOrigin(origins = "http://localhost:3000")
public class QuestionController {

    @Autowired
    private QuestionRepository questionRepository;

    @PostMapping("/submit-questions")
    public String submitQuestions(@RequestBody List<Question> questions) {
        questionRepository.saveAll(questions);
        return "Questions received";
    }

    @GetMapping("/questions")
    public List<Question> getQuestions() {
        return questionRepository.findAll();
    }
}
