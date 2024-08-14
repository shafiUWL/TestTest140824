// AnswerRequest.java: This data transfer object (DTO) represents a request 
// containing a student's answer and the corresponding question.
// It includes fields for the question and answer, with getters and setters 
// for each.



package com.education.ai.assessmentmarker.dto;

public class AnswerRequest {
    private String question;
    private String answer;

    // Getters and setters
    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public String getAnswer() {
        return answer;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }
}
