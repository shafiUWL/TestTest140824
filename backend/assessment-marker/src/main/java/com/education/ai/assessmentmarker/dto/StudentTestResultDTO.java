// StudentTestResultDTO.java: This data transfer object (DTO) represents a 
// student's test result details.
// It includes fields for student ID, test name, score, max score, and grade, 
// with getters and setters for each.



package com.education.ai.assessmentmarker.dto;

public class StudentTestResultDTO {
    private String studentId;
    private String testName;
    private int score;
    private int maxScore;
    private String grade;

    // Getters and setters
    public String getStudentId() {
        return studentId;
    }

    public void setStudentId(String studentId) {
        this.studentId = studentId;
    }

    public String getTestName() {
        return testName;
    }

    public void setTestName(String testName) {
        this.testName = testName;
    }

    public int getScore() {
        return score;
    }

    public void setScore(int score) {
        this.score = score;
    }

    public int getMaxScore() {
        return maxScore;
    }

    public void setMaxScore(int maxScore) {
        this.maxScore = maxScore;
    }

    public String getGrade() {
        return grade;
    }

    public void setGrade(String grade) {
        this.grade = grade;
    }
}
