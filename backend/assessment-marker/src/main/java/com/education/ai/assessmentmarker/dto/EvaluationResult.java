// EvaluationResult.java: This data transfer object (DTO) represents the 
// result of an AI evaluation.
// It includes a single field for the result, with a getter and setter.


package com.education.ai.assessmentmarker.dto;

public class EvaluationResult {
    private String result;

    public EvaluationResult(String result) {
        this.result = result;
    }

    // Getter
    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }
}
