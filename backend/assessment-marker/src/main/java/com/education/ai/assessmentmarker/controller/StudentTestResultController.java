// StudentTestResultController.java: This controller class handles HTTP requests 
// related to student test results.
// It provides endpoints to save test results and retrieve results by student ID.
// The endpoint "/api/results" accepts POST requests to save test results.
// The endpoint "/api/results/{studentId}" retrieves a list of test results for 
// a specific student.



package com.education.ai.assessmentmarker.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import com.education.ai.assessmentmarker.StudentTestResult;
import com.education.ai.assessmentmarker.service.StudentTestResultService;
import java.util.List;

@RestController
@RequestMapping("/api/results")
@CrossOrigin(origins = "*")
public class StudentTestResultController {

    @Autowired
    private StudentTestResultService service;

    @PostMapping
    public StudentTestResult saveResult(@RequestBody StudentTestResult result) {
        return service.saveResult(result);
    }

    @GetMapping("/{studentId}")
    public List<StudentTestResult> getResultsByStudentId(@PathVariable String studentId) {
        return service.getResultsByStudentId(studentId);
    }
}
