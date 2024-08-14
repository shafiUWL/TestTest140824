package com.education.ai.assessmentmarker.service;

import org.springframework.stereotype.Service;

import com.education.ai.assessmentmarker.dto.AnswerRequest;
import com.education.ai.assessmentmarker.dto.EvaluationResult;

@Service
public class AIEvaluationService {

    public EvaluationResult evaluate(AnswerRequest answerRequest) {
        // Placeholder logic for AI evaluation
        String result = "This is a placeholder evaluation result.";
        return new EvaluationResult(result);
    }
}
