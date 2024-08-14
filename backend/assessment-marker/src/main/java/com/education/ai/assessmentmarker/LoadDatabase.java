// LoadDatabase.java: This configuration class initializes the database with 
// initial data.
// It uses CommandLineRunner to check for existing data and saves default 
// questions if none exist.




package com.education.ai.assessmentmarker;

import com.education.ai.assessmentmarker.model.Question;
import com.education.ai.assessmentmarker.repository.QuestionRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class LoadDatabase {

    @Bean
    CommandLineRunner initDatabase(QuestionRepository repository) {
        return args -> {
            // Check if data already exists
            List<Question> existingQuestions = repository.findAll();
            if (existingQuestions.isEmpty()) {
                repository.save(new Question("What is the capital of France?", "Paris"));
                repository.save(new Question("What is 2 + 2?", "4"));
            }
        };
    }
}
