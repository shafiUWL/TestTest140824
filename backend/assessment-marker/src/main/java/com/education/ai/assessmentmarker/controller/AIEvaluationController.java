// AIEvaluationController.java: This controller class handles HTTP requests 
// for evaluating student answers.
// It autowires the AIEvaluationService to perform the evaluation logic.
// The endpoint "/evaluate" accepts POST requests with an AnswerRequest body 
// and returns an EvaluationResult.


package com.education.ai.assessmentmarker.controller;

import com.education.ai.assessmentmarker.dto.AnswerRequest;
import com.education.ai.assessmentmarker.dto.EvaluationResult;
import com.education.ai.assessmentmarker.service.AIEvaluationService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class AIEvaluationController {

    @Autowired
    private AIEvaluationService aiEvaluationService;

    @PostMapping("/evaluate")
    public EvaluationResult evaluateAnswer(@RequestBody AnswerRequest answerRequest) {
        return aiEvaluationService.evaluate(answerRequest);
    }
}
