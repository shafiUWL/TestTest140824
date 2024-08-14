from googleapiclient.discovery import build
import os
import pandas as pd
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
from collections import OrderedDict
import requests

# Initialize Flask app
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing for the app
CORS(app)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Directory for CSV files
csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# Metadata file for CSV files
metadata_file = 'csv_files_metadata.json'

# Add your Google Custom Search API key and CSE ID
api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
cse_id = 'f04f843d4b95b457f'

# Function to perform Google search
def google_search(query, api_key, cse_id):
    service = build("customsearch", "v1", developerKey=api_key)
    # Fetch 3 results to aggregate context
    res = service.cse().list(q=query, cx=cse_id, num=3).execute()
    return res

# Function to check if a question is mathematical
def is_mathematical_question(question):
    math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
    if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
        return True
    return False

# Function to solve mathematical questions
def solve_math_question(question):
    api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.text
    return "Could not solve the mathematical question."

# Function to load metadata
def load_metadata():
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as file:
            return json.load(file)
    return {}

# Function to save metadata
def save_metadata(metadata):
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file, indent=4)

# Function to load CSV files and their content
def load_csv_files():
    subjects = {}
    for filename in os.listdir(csv_directory):
        if filename.endswith('.csv'):
            subject_name = filename.split('.')[0].lower()
            df = pd.read_csv(os.path.join(csv_directory, filename))
            print(f"Loading CSV file: {filename}")
            if 'order' in df.columns:
                df = df.sort_values(by=['order'])
                subjects[subject_name] = df.set_index('question').to_dict(orient='index')
            else:
                raise KeyError(f"'order' column is missing in the CSV file: {filename}")
    print(f"Loaded subjects: {subjects.keys()}")
    return subjects

# Load correct answers from CSV files
correct_answers = load_csv_files()
print(f"Correct answers loaded: {correct_answers.keys()}")
# Set similarity threshold for answer evaluation
question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
similarity_threshold = 0.65

# Function to calculate grade based on score
def calculate_grade(score, max_score):
    percentage = (score / max_score) * 100
    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'

# Endpoint to generate text based on a prompt
@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
        outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
        generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({'generated_text': generated})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Function to generate feedback based on the score and results
def generate_feedback(score, max_score, grade, results):
    incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
    feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
    if incorrect_topics:
        feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
    else:
        feedback_prompt += "Excellent work, no areas of improvement."

    inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
    outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
    comment = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Log generated comment for debugging
    print(f"Generated comment: {comment}")
    print(f"The generated comments need to be fixed....")

    return comment

# Endpoint to handle question answering
@app.route('/qa', methods=['POST'])
def qa():
    try:
        data = request.json
        question = data['question']
        
        if is_mathematical_question(question):
            answer = solve_math_question(question)
            return jsonify({'answer': answer})
        
        search_response = google_search(question, api_key, cse_id)
        if not search_response.get('items'):
            return jsonify({'answer': 'Could not find an answer on the web.'})

        context = ""
        for item in search_response['items']:
            context += item['snippet'] + " "
            if len(context) > 1000:
                break

        result = qa_pipeline(question=question, context=context)
        print(f"QA Result: {result}")

        return jsonify({'answer': result['answer']})
    except Exception as e:
        # Log error during question answering
        print(f"Error during question answering: {e}")
        return jsonify({'error': str(e)}), 400

# Endpoint to evaluate student answers
@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        # Log received data
        print(f"Received data: {data}")
        subject = data.get('subject').lower()
        answers = data.get('answers')
        student_id = data.get('student_id')
        results = OrderedDict()
        total_score = 0
        max_score = sum(question_marks[subject].values())

        for key, value in answers.items():
            correct_answer = correct_answers[subject][key]['answer']
            similarity_check = correct_answers[subject][key]['similarity_check']
            keywords = str(correct_answers[subject][key]['keywords']).split(',')
            order = correct_answers[subject][key]['order']

            if value == correct_answer:
                results[key] = {
                    'order': order,
                    'student_answer': value,
                    'correct_answer': correct_answer,
                    'feedback': "Your answer is correct.",
                    'similarity': 1.0,
                    'marks': question_marks[subject][key]
                }
                total_score += question_marks[subject][key]
            else:
                if similarity_check:
                    sentences = [value, correct_answer]
                    embeddings = sentence_model.encode(sentences)
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
                    )
                    similarity_score = cosine_sim.item()
                    if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
                        results[key] = {
                            'order': order,
                            'student_answer': value,
                            'correct_answer': correct_answer,
                            'feedback': "Your answer is partially correct.",
                            'similarity': similarity_score,
                            'marks': question_marks[subject][key] // 2  # Use floor division to avoid partial marks
                        }
                        total_score += question_marks[subject][key] // 2
                    else:
                        results[key] = {
                            'order': order,
                            'student_answer': value,
                            'correct_answer': correct_answer,
                            'feedback': "Your answer is incorrect.",
                            'similarity': similarity_score,
                            'marks': 0
                        }
                else:
                    results[key] = {
                        'order': order,
                        'student_answer': value,
                        'correct_answer': correct_answer,
                        'feedback': "Your answer is incorrect.",
                        'similarity': 0,
                        'marks': 0
                    }

        grade = calculate_grade(total_score, max_score)
        comment = generate_feedback(total_score, max_score, grade, results)
        # Log generated comment
        print(f"Generated comment: {comment}")

        result_data = {
            'studentId': student_id,
            'testName': subject,
            'score': total_score,
            'maxScore': max_score,
            'grade': grade,
            'comment': comment  # Ensure comment is included in the response
        }
        response = requests.post('http://localhost:8080/api/results', json=result_data)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to save test result'}), 500

        return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'comment': comment})
    except Exception as e:
        # Log error
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Endpoint to upload a new CSV file
@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        test_name = request.form['testName'].lower()
        timestamp = datetime.now().isoformat()
        file.save(os.path.join(csv_directory, f"{test_name}.csv"))

        metadata = load_metadata()
        metadata[test_name] = timestamp
        save_metadata(metadata)

        global correct_answers, question_marks
        correct_answers = load_csv_files()
        question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

        return jsonify({'message': 'CSV file uploaded successfully.'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint to delete a test
@app.route('/delete-test', methods=['POST'])
def delete_test():
    try:
        test_name = request.json.get('testName').lower()
        file_path = os.path.join(csv_directory, f"{test_name}.csv")

        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            return jsonify({'message': 'File not found.'}), 404

        metadata = load_metadata()
        if test_name in metadata:
            del metadata[test_name]
            save_metadata(metadata)
        else:
            return jsonify({'message': 'Test not found in metadata.'}), 404

        global correct_answers, question_marks
        correct_answers = load_csv_files()
        question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

        return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint to get metadata of all tests
@app.route('/get-tests', methods=['GET'])
def get_tests():
    try:
        metadata = load_metadata()
        return jsonify(metadata), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint to get questions of a specific subject
@app.route('/get-questions/<subject>', methods=['GET'])
def get_questions(subject):
    try:
        subject = subject.lower()
        # Log fetching questions for the subject
        print(f"Fetching questions for subject: {subject}")
        if subject in correct_answers:
            questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
            questions = sorted(questions, key=lambda x: x['order'])
            # Log questions for the subject
            print(f"Questions for {subject}: {questions}")
            return jsonify(questions), 200
        else:
            # Log subject not found
            print(f"Subject not found: {subject}")
            return jsonify({'error': 'Subject not found'}), 404
    except Exception as e:
        # Log error fetching questions for the subject
        print(f"Error fetching questions for {subject}: {str(e)}")
        return jsonify({'error': str(e)}), 400

# Endpoint to list all files in the directory
@app.route('/list-files', methods=['GET'])
def list_files():
    try:
        files = list_files_in_directory()
        return jsonify({'files': files}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Function to list all files in the directory
def list_files_in_directory():
    try:
        files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f)) and not f.startswith('.')]
        return files
    except Exception as e:
        return []

if __name__ == '__main__':
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    # Run the Flask app on port 5001
    app.run(host='0.0.0.0', port=5001)





























# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def generate_feedback(score, max_score, grade, results):
#     incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
#     feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
#     if incorrect_topics:
#         feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
#     else:
#         feedback_prompt += "Excellent work, no areas of improvement."

#     inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
#     outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#     comment = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

#     print(f"Generated comment: {comment}")  # Log generated comment for debugging

#     return comment



# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:
#                 break

#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")

#         return jsonify({'answer': result['answer']})
#     except Exception as e:
#         print(f"Error during question answering: {e}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         student_id = data.get('student_id')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key] // 2  # Use floor division to avoid partial marks
#                         }
#                         total_score += question_marks[subject][key] // 2
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)
#         comment = generate_feedback(total_score, max_score, grade, results)
#         print(f"Generated comment: {comment}")  # Log generated comment

#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade,
#             'comment': comment  # Ensure comment is included in the response
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'comment': comment})
#     except Exception as e:
#         print(f"Error: {str(e)}")  # Log error
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f)) and not f.startswith('.')]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)




















# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def generate_feedback(score, max_score, grade, results):
#     incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
#     feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
#     if incorrect_topics:
#         feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
#     else:
#         feedback_prompt += "Excellent work, no areas of improvement."

#     inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
#     outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#     comment = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

#     print(f"Generated comment: {comment}")  # Log generated comment for debugging

#     return comment



# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:
#                 break

#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")

#         return jsonify({'answer': result['answer']})
#     except Exception as e:
#         print(f"Error during question answering: {e}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         student_id = data.get('student_id')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key] // 2  # Use floor division to avoid partial marks
#                         }
#                         total_score += question_marks[subject][key] // 2
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)
#         comment = generate_feedback(total_score, max_score, grade, results)
#         print(f"Generated comment: {comment}")  # Log generated comment

#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade,
#             'comment': comment  # Ensure comment is included in the response
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'comment': comment})
#     except Exception as e:
#         print(f"Error: {str(e)}")  # Log error
#         return jsonify({'error': str(e)}), 400


# # @app.route('/evaluate', methods=['POST'])
# # def evaluate():
# #     try:
# #         data = request.json
# #         print(f"Received data: {data}")
# #         subject = data.get('subject').lower()
# #         answers = data.get('answers')
# #         student_id = data.get('student_id')
# #         results = OrderedDict()
# #         total_score = 0
# #         max_score = sum(question_marks[subject].values())

# #         for key, value in answers.items():
# #             correct_answer = correct_answers[subject][key]['answer']
# #             similarity_check = correct_answers[subject][key]['similarity_check']
# #             keywords = str(correct_answers[subject][key]['keywords']).split(',')
# #             order = correct_answers[subject][key]['order']

# #             if value == correct_answer:
# #                 results[key] = {
# #                     'order': order,
# #                     'student_answer': value,
# #                     'correct_answer': correct_answer,
# #                     'feedback': "Your answer is correct.",
# #                     'similarity': 1.0,
# #                     'marks': question_marks[subject][key]
# #                 }
# #                 total_score += question_marks[subject][key]
# #             else:
# #                 if similarity_check:
# #                     sentences = [value, correct_answer]
# #                     embeddings = sentence_model.encode(sentences)
# #                     cosine_sim = torch.nn.functional.cosine_similarity(
# #                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
# #                     )
# #                     similarity_score = cosine_sim.item()
# #                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is partially correct.",
# #                             'similarity': similarity_score,
# #                             'marks': question_marks[subject][key] // 2  # Use floor division to avoid partial marks
# #                         }
# #                         total_score += question_marks[subject][key] // 2
# #                     else:
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is incorrect.",
# #                             'similarity': similarity_score,
# #                             'marks': 0
# #                         }
# #                 else:
# #                     results[key] = {
# #                         'order': order,
# #                         'student_answer': value,
# #                         'correct_answer': correct_answer,
# #                         'feedback': "Your answer is incorrect.",
# #                         'similarity': 0,
# #                         'marks': 0
# #                     }

# #         grade = calculate_grade(total_score, max_score)
# #         comment = generate_feedback(total_score, max_score, grade, results)
# #         print(f"Generated comment: {comment}")  # Log generated comment

# #         result_data = {
# #             'studentId': student_id,
# #             'testName': subject,
# #             'score': total_score,
# #             'maxScore': max_score,
# #             'grade': grade,
# #             'comment': comment  # Ensure comment is included in the response
# #         }
# #         response = requests.post('http://localhost:8080/api/results', json=result_data)
# #         if response.status_code != 200:
# #             return jsonify({'error': 'Failed to save test result'}), 500

# #         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'comment': comment})
# #     except Exception as e:
# #         print(f"Error: {str(e)}")  # Log error
# #         return jsonify({'error': str(e)}), 400


# # @app.route('/evaluate', methods=['POST'])
# # def evaluate():
# #     try:
# #         data = request.json
# #         print(f"Received data: {data}")
# #         subject = data.get('subject').lower()
# #         answers = data.get('answers')
# #         student_id = data.get('student_id')
# #         results = OrderedDict()
# #         total_score = 0
# #         max_score = sum(question_marks[subject].values())

# #         for key, value in answers.items():
# #             correct_answer = correct_answers[subject][key]['answer']
# #             similarity_check = correct_answers[subject][key]['similarity_check']
# #             keywords = str(correct_answers[subject][key]['keywords']).split(',')
# #             order = correct_answers[subject][key]['order']

# #             if value == correct_answer:
# #                 results[key] = {
# #                     'order': order,
# #                     'student_answer': value,
# #                     'correct_answer': correct_answer,
# #                     'feedback': "Your answer is correct.",
# #                     'similarity': 1.0,
# #                     'marks': question_marks[subject][key]
# #                 }
# #                 total_score += question_marks[subject][key]
# #             else:
# #                 if similarity_check:
# #                     sentences = [value, correct_answer]
# #                     embeddings = sentence_model.encode(sentences)
# #                     cosine_sim = torch.nn.functional.cosine_similarity(
# #                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
# #                     )
# #                     similarity_score = cosine_sim.item()
# #                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is partially correct.",
# #                             'similarity': similarity_score,
# #                             'marks': question_marks[subject][key] // 2  # Use floor division to avoid partial marks
# #                         }
# #                         total_score += question_marks[subject][key] // 2
# #                     else:
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is incorrect.",
# #                             'similarity': similarity_score,
# #                             'marks': 0
# #                         }
# #                 else:
# #                     results[key] = {
# #                         'order': order,
# #                         'student_answer': value,
# #                         'correct_answer': correct_answer,
# #                         'feedback': "Your answer is incorrect.",
# #                         'similarity': 0,
# #                         'marks': 0
# #                     }

# #         grade = calculate_grade(total_score, max_score)
# #         comment = generate_feedback(total_score, max_score, grade, results)
# #         print(f"Generated comment: {comment}")  # Log generated comment

# #         result_data = {
# #             'studentId': student_id,
# #             'testName': subject,
# #             'score': total_score,
# #             'maxScore': max_score,
# #             'grade': grade,
# #             'comment': comment  # Ensure comment is included in the response
# #         }
# #         response = requests.post('http://localhost:8080/api/results', json=result_data)
# #         if response.status_code != 200:
# #             return jsonify({'error': 'Failed to save test result'}), 500

# #         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'comment': comment})
# #     except Exception as e:
# #         print(f"Error: {str(e)}")  # Log error
# #         return jsonify({'error': str(e)}), 400



# def generate_feedback(score, max_score, grade, results):
#     incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
#     feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
#     if incorrect_topics:
#         feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
#     else:
#         feedback_prompt += "Excellent work, no areas of improvement."

#     print(f"Feedback Prompt: {feedback_prompt}")  # Logging feedback prompt for debugging

#     inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
#     outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#     comment = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     print(f"Generated Comment: {comment}")  # Logging generated comment
    
#     return comment



# # @app.route('/evaluate', methods=['POST'])
# # def evaluate():
# #     try:
# #         data = request.json
# #         print(f"Received data: {data}")
# #         subject = data.get('subject').lower()
# #         answers = data.get('answers')
# #         student_id = data.get('student_id')
# #         results = OrderedDict()
# #         total_score = 0
# #         max_score = sum(question_marks[subject].values())

# #         for key, value in answers.items():
# #             correct_answer = correct_answers[subject][key]['answer']
# #             similarity_check = correct_answers[subject][key]['similarity_check']
# #             keywords = str(correct_answers[subject][key]['keywords']).split(',')
# #             order = correct_answers[subject][key]['order']

# #             if value == correct_answer:
# #                 results[key] = {
# #                     'order': order,
# #                     'student_answer': value,
# #                     'correct_answer': correct_answer,
# #                     'feedback': "Your answer is correct.",
# #                     'similarity': 1.0,
# #                     'marks': question_marks[subject][key]
# #                 }
# #                 total_score += question_marks[subject][key]
# #             else:
# #                 if similarity_check:
# #                     sentences = [value, correct_answer]
# #                     embeddings = sentence_model.encode(sentences)
# #                     cosine_sim = torch.nn.functional.cosine_similarity(
# #                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
# #                     )
# #                     similarity_score = cosine_sim.item()
# #                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is partially correct.",
# #                             'similarity': similarity_score,
# #                             'marks': question_marks[subject][key] // 2  # Use floor division to avoid partial marks
# #                         }
# #                         total_score += question_marks[subject][key] // 2
# #                     else:
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is incorrect.",
# #                             'similarity': similarity_score,
# #                             'marks': 0
# #                         }
# #                 else:
# #                     results[key] = {
# #                         'order': order,
# #                         'student_answer': value,
# #                         'correct_answer': correct_answer,
# #                         'feedback': "Your answer is incorrect.",
# #                         'similarity': 0,
# #                         'marks': 0
# #                     }

# #         grade = calculate_grade(total_score, max_score)
# #         feedback = generate_feedback(total_score, max_score, grade, results)
# #         print(f"Generated feedback: {feedback}")  # Log generated feedback

# #         result_data = {
# #             'studentId': student_id,
# #             'testName': subject,
# #             'score': total_score,
# #             'maxScore': max_score,
# #             'grade': grade,
# #             'feedback': feedback  # Ensure feedback is included in the response
# #         }
# #         response = requests.post('http://localhost:8080/api/results', json=result_data)
# #         if response.status_code != 200:
# #             return jsonify({'error': 'Failed to save test result'}), 500

# #         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'feedback': feedback})
# #     except Exception as e:
# #         print(f"Error: {str(e)}")  # Log error
# #         return jsonify({'error': str(e)}), 400



# # @app.route('/evaluate', methods=['POST'])
# # def evaluate():
# #     try:
# #         data = request.json
# #         print(f"Received data: {data}")
# #         subject = data.get('subject').lower()
# #         answers = data.get('answers')
# #         student_id = data.get('student_id')
# #         results = OrderedDict()
# #         total_score = 0
# #         max_score = sum(question_marks[subject].values())

# #         for key, value in answers.items():
# #             correct_answer = correct_answers[subject][key]['answer']
# #             similarity_check = correct_answers[subject][key]['similarity_check']
# #             keywords = str(correct_answers[subject][key]['keywords']).split(',')
# #             order = correct_answers[subject][key]['order']

# #             if value == correct_answer:
# #                 results[key] = {
# #                     'order': order,
# #                     'student_answer': value,
# #                     'correct_answer': correct_answer,
# #                     'feedback': "Your answer is correct.",
# #                     'similarity': 1.0,
# #                     'marks': question_marks[subject][key]
# #                 }
# #                 total_score += question_marks[subject][key]
# #             else:
# #                 if similarity_check:
# #                     sentences = [value, correct_answer]
# #                     embeddings = sentence_model.encode(sentences)
# #                     cosine_sim = torch.nn.functional.cosine_similarity(
# #                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
# #                     )
# #                     similarity_score = cosine_sim.item()
# #                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is partially correct.",
# #                             'similarity': similarity_score,
# #                             'marks': question_marks[subject][key] // 2  # Use floor division to avoid partial marks
# #                         }
# #                         total_score += question_marks[subject][key] // 2
# #                     else:
# #                         results[key] = {
# #                             'order': order,
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is incorrect.",
# #                             'similarity': similarity_score,
# #                             'marks': 0
# #                         }
# #                 else:
# #                     results[key] = {
# #                         'order': order,
# #                         'student_answer': value,
# #                         'correct_answer': correct_answer,
# #                         'feedback': "Your answer is incorrect.",
# #                         'similarity': 0,
# #                         'marks': 0
# #                     }

# #         grade = calculate_grade(total_score, max_score)
# #         feedback = generate_feedback(total_score, max_score, grade, results)
# #         print(f"Calculated grade: {grade}")

# #         result_data = {
# #             'studentId': student_id,
# #             'testName': subject,
# #             'score': total_score,
# #             'maxScore': max_score,
# #             'grade': grade
# #         }
# #         response = requests.post('http://localhost:8080/api/results', json=result_data)
# #         if response.status_code != 200:
# #             return jsonify({'error': 'Failed to save test result'}), 500

# #         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'feedback': feedback})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# # @app.route('/delete-test', methods=['POST'])
# # def delete_test():
# #     try:
# #         test_name = request.json.get('testName').lower()
# #         file_path = os.path.join(csv_directory, f"{test_name}.csv")

# #         if os.path.exists(file_path):
# #             os.remove(file_path)
# #         else:
# #             return jsonify({'message': 'File not found.'}), 404

# #         metadata = load_metadata()
# #         if test_name in metadata:
# #             del metadata[test_name]
# #             save_metadata(metadata)
# #         else:
# #             return jsonify({'message': 'Test not found in metadata.'}), 404

# #         global correct_answers, question_marks
# #         correct_answers = load_csv_files()
# #         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

# #         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# # @app.route('/list-files', methods=['GET'])
# # def list_files():
# #     try:
# #         files = list_files_in_directory()
# #         return jsonify({'files': files}), 200
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f)) and not f.startswith('.')]
#         return files
#     except Exception as e:
#         return []


# # def list_files_in_directory():
# #     try:
# #         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
# #         return files
# #     except Exception as e:
# #         return []
    
# def generate_feedback(score, max_score, grade, results):
#     incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
#     feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
#     if incorrect_topics:
#         feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
#     else:
#         feedback_prompt += "Excellent work, no areas of improvement."

#     print(f"Feedback Prompt: {feedback_prompt}")  # Logging feedback prompt for debugging

#     inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
#     outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#     feedback = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     print(f"Generated Feedback: {feedback}")  # Logging generated feedback for debugging
    
#     return feedback


# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


















# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# def generate_feedback(score, max_score, grade, results):
#     incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
#     feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
#     if incorrect_topics:
#         feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
#     else:
#         feedback_prompt += "Excellent work, no areas of improvement."

#     inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
#     outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#     feedback = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return feedback

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:
#                 break

#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")

#         return jsonify({'answer': result['answer']})
#     except Exception as e:
#         print(f"Error during question answering: {e}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         student_id = data.get('student_id')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Changed to full marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)
#         feedback = generate_feedback(total_score, max_score, grade, results)
#         print(f"Calculated grade: {grade}")

#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'feedback': feedback})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# def generate_feedback(score, max_score, grade, results):
#     incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
#     feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
#     if incorrect_topics:
#         feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
#     else:
#         feedback_prompt += "Excellent work, no areas of improvement."

#     inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
#     outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#     feedback = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return feedback

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         return jsonify({'answer': result['answer']})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key] / 2  # Partial marks for similar answers
#                         }
#                         total_score += question_marks[subject][key] / 2
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)
#         feedback = generate_feedback(total_score, max_score, grade, results)
#         print(f"Calculated grade: {grade}")  # Debugging line

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'feedback': feedback})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# def generate_feedback(score, max_score, grade, results):
#     incorrect_topics = [feedback['correct_answer'] for question, feedback in results.items() if feedback['marks'] == 0]
#     feedback_prompt = f"Student scored {score} out of {max_score}, receiving a grade of {grade}. "
#     if incorrect_topics:
#         feedback_prompt += f"They need to work on the following topics: {', '.join(incorrect_topics)}."
#     else:
#         feedback_prompt += "Excellent work, no areas of improvement."

#     inputs = t5_tokenizer.encode(feedback_prompt, return_tensors='pt')
#     outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#     feedback = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return feedback

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         return jsonify({'answer': result['answer']})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key] / 2  # Partial marks for similar answers
#                         }
#                         total_score += question_marks[subject][key] / 2
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)
#         feedback = generate_feedback(total_score, max_score, grade, results)
#         print(f"Calculated grade: {grade}")  # Debugging line

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade, 'feedback': feedback})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         return jsonify({'answer': result['answer']})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)
#         print(f"Calculated grade: {grade}")  # Debugging line

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         # Generate a complete response using T5
#         prompt = f"Based on the context, the answer to the question '{question}' is '{result['answer']}'. Provide a detailed explanation."
#         inputs = t5_tokenizer.encode(prompt, return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=200, num_return_sequences=1, do_sample=True, temperature=1.0, top_p=0.95)
#         generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         return jsonify({'answer': generated_text})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         # Generate a complete response using T5
#         prompt = f"The answer to the question '{question}' is {result['answer']}. Explain in detail."
#         inputs = t5_tokenizer.encode(prompt, return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
#         generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         return jsonify({'answer': generated_text})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# from googleapiclient.discovery import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         # Generate a complete response using T5
#         prompt = f"The answer to the question '{question}' is {result['answer']}. Explain in detail."
#         inputs = t5_tokenizer.encode(prompt, return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
#         generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         return jsonify({'answer': generated_text})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# from distutils.command import build
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."



# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for question, student_answer in answers.items():
#             inputs = tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
#             outputs = model(**inputs)
#             prediction = torch.argmax(outputs.logits, dim=-1).item()

#             correct_answer = correct_answers[subject][question]['answer']
#             similarity_check = correct_answers[subject][question]['similarity_check']
#             keywords = str(correct_answers[subject][question]['keywords']).split(',')
#             order = correct_answers[subject][question]['order']
#             marks = question_marks[subject][question]

#             if prediction == 0:  # Assuming 0 is the label for correct answers
#                 total_score += marks
#                 results[question] = {
#                     'order': order,
#                     'student_answer': student_answer,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': marks
#                 }
#             else:
#                 if similarity_check:
#                     sentences = [student_answer, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in student_answer for keyword in keywords):
#                         total_score += marks / 2  # Partial credit for similar answers
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': marks / 2
#                         }
#                     else:
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[question] = {
#                         'order': order,
#                         'student_answer': student_answer,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)



# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests
# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for question, student_answer in answers.items():
#             inputs = tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
#             outputs = model(**inputs)
#             prediction = torch.argmax(outputs.logits, dim=-1).item()

#             correct_answer = correct_answers[subject][question]['answer']
#             similarity_check = correct_answers[subject][question]['similarity_check']
#             keywords = str(correct_answers[subject][question]['keywords']).split(',')
#             order = correct_answers[subject][question]['order']
#             marks = question_marks[subject][question]

#             if prediction == 0:  # Assuming 0 is the label for correct answers
#                 total_score += marks
#                 results[question] = {
#                     'order': order,
#                     'student_answer': student_answer,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': marks
#                 }
#             else:
#                 if similarity_check:
#                     sentences = [student_answer, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in student_answer for keyword in keywords):
#                         total_score += marks / 2  # Partial credit for similar answers
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': marks / 2
#                         }
#                     else:
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[question] = {
#                         'order': order,
#                         'student_answer': student_answer,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             # Ensure keywords is treated as a string
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)










# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests
# from googleapiclient.discovery import build

# app = Flask(__name__)
# CORS(app)

# # Load models
# bert_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bert")
# bert_model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
# sentence_model = SentenceTransformer('fine_tuned_sentence_bert')
# t5_tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5')
# t5_model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5')
# qa_pipeline = pipeline("question-answering", model="fine_tuned_qa")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res


# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         # Generate a complete response using T5
#         prompt = f"The answer to the question '{question}' is {result['answer']}. Explain in detail."
#         inputs = t5_tokenizer.encode(prompt, return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
#         generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         return jsonify({'answer': generated_text})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for question, student_answer in answers.items():
#             inputs = bert_tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
#             outputs = bert_model(**inputs)
#             prediction = torch.argmax(outputs.logits, dim=-1).item()

#             correct_answer = correct_answers[subject][question]['answer']
#             similarity_check = correct_answers[subject][question]['similarity_check']
#             keywords = str(correct_answers[subject][question]['keywords']).split(',')
#             order = correct_answers[subject][question]['order']
#             marks = question_marks[subject][question]

#             if prediction == 0:  # Assuming 0 is the label for correct answers
#                 total_score += marks
#                 results[question] = {
#                     'order': order,
#                     'student_answer': student_answer,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': marks
#                 }
#             else:
#                 if similarity_check:
#                     sentences = [student_answer, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in student_answer for keyword in keywords):
#                         total_score += marks / 2  # Partial credit for similar answers
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': marks / 2
#                         }
#                     else:
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[question] = {
#                         'order': order,
#                         'student_answer': student_answer,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key] / 2  # Partial marks for similar answers
#                         }
#                         total_score += question_marks[subject][key] / 2
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)
#         print(f"Calculated grade: {grade}")  # Debugging line

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         print(f"Received prompt for generation: {data['prompt']}")  # Debugging line
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"Generated text: {generated}")  # Debugging line
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         print(f"Error during text generation: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400
    

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# # import os
# # import pandas as pd
# # import json
# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
# # from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# # from sentence_transformers import SentenceTransformer
# # import torch
# # from datetime import datetime
# # from collections import OrderedDict
# # import requests
# # # import wikipediaapi
# # from googleapiclient.discovery import build
# # import re
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests
# from googleapiclient.discovery import build

# app = Flask(__name__)
# CORS(app)

# # # Load the QA pipeline
# # qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# # # Load tokenizer and model
# # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# # model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# # sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# # t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')  # Use 't5-base' instead of 't5-small'
# # t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# # Load models
# # tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bert")
# # model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
# # sentence_model = SentenceTransformer('fine_tuned_sentence_bert')
# # t5_model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5')
# # t5_tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5')
# # qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# # # Load models
# # tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bert")
# # model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
# # sentence_model = SentenceTransformer('fine_tuned_sentence_bert')
# # t5_model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5')
# # t5_tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5')
# # qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# # Load models
# bert_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_bert")
# bert_model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
# sentence_model = SentenceTransformer('fine_tuned_sentence_bert')
# t5_tokenizer = T5Tokenizer.from_pretrained('fine_tuned_t5')
# t5_model = T5ForConditionalGeneration.from_pretrained('fine_tuned_t5')
# qa_pipeline = pipeline("question-answering", model="fine_tuned_qa")

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'


# # Add your Google Custom Search API key and CSE ID
# api_key = 'AIzaSyB4r8tNZOkopjeLSrVL5TN8AkUpkjF3NsQ'
# cse_id = 'f04f843d4b95b457f'

# def google_search(query, api_key, cse_id):
#     service = build("customsearch", "v1", developerKey=api_key)
#     res = service.cse().list(q=query, cx=cse_id, num=3).execute()  # Fetch 3 results to aggregate context
#     return res

# # Load the QA pipeline
# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# @app.route('/qa', methods=['POST'])
# def qa():
#     try:
#         data = request.json
#         question = data['question']
        
#         # Check if the question is mathematical
#         if is_mathematical_question(question):
#             answer = solve_math_question(question)
#             return jsonify({'answer': answer})
        
#         # Fetch context from Google Custom Search for non-mathematical questions
#         search_response = google_search(question, api_key, cse_id)
#         if not search_response.get('items'):
#             return jsonify({'answer': 'Could not find an answer on the web.'})

#         # Aggregate snippets from multiple search results
#         context = ""
#         for item in search_response['items']:
#             context += item['snippet'] + " "
#             if len(context) > 1000:  # Limit context length to 1000 characters
#                 break

#         # Use the context with the QA pipeline
#         result = qa_pipeline(question=question, context=context)
#         print(f"QA Result: {result}")  # Debugging line

#         # Generate a complete response using T5
#         prompt = f"The answer to the question '{question}' is {result['answer']}. Explain in detail."
#         inputs = t5_tokenizer.encode(prompt, return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
#         generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         return jsonify({'answer': generated_text})
#     except Exception as e:
#         print(f"Error during question answering: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# # @app.route('/qa', methods=['POST'])
# # def qa():
# #     try:
# #         data = request.json
# #         question = data['question']
        
# #         # Check if the question is mathematical
# #         if is_mathematical_question(question):
# #             answer = solve_math_question(question)
# #             return jsonify({'answer': answer})
        
# #         # Fetch context from Google Custom Search for non-mathematical questions
# #         search_response = google_search(question, api_key, cse_id)
# #         if not search_response.get('items'):
# #             return jsonify({'answer': 'Could not find an answer on the web.'})

# #         # Aggregate snippets from multiple search results
# #         context = ""
# #         for item in search_response['items']:
# #             context += item['snippet'] + " "
# #             if len(context) > 1000:  # Limit context length to 1000 characters
# #                 break

# #         # Use the context with the QA pipeline
# #         result = qa_pipeline(question=question, context=context)
# #         print(f"QA Result: {result}")  # Debugging line

# #         # Generate a complete response using T5
# #         prompt = f"The answer to the question '{question}' is {result['answer']}. Explain in detail."
# #         inputs = t5_tokenizer.encode(prompt, return_tensors='pt')
# #         outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
# #         generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
# #         return jsonify({'answer': generated_text})
# #     except Exception as e:
# #         print(f"Error during question answering: {e}")  # Debugging line
# #         return jsonify({'error': str(e)}), 400


# # @app.route('/qa', methods=['POST'])
# # def qa():
# #     try:
# #         data = request.json
# #         question = data['question']
        
# #         # Check if the question is mathematical
# #         if is_mathematical_question(question):
# #             answer = solve_math_question(question)
# #             return jsonify({'answer': answer})
        
# #         # Fetch context from Google Custom Search for non-mathematical questions
# #         search_response = google_search(question, api_key, cse_id)
# #         if not search_response.get('items'):
# #             return jsonify({'answer': 'Could not find an answer on the web.'})

# #         # Aggregate snippets from multiple search results
# #         context = ""
# #         for item in search_response['items']:
# #             context += item['snippet'] + " "
# #             if len(context) > 1000:  # Limit context length to 1000 characters
# #                 break

# #         # Use the context with the QA pipeline
# #         result = qa_pipeline(question=question, context=context)
# #         print(f"QA Result: {result}")  # Debugging line

# #         # Generate a complete response using T5
# #         prompt = f"The answer to the question '{question}' is {result['answer']}. Explain in detail."
# #         inputs = t5_tokenizer.encode(prompt, return_tensors='pt')
# #         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
# #         generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
# #         return jsonify({'answer': generated_text})
# #     except Exception as e:
# #         print(f"Error during question answering: {e}")  # Debugging line
# #         return jsonify({'error': str(e)}), 400

# # @app.route('/qa', methods=['POST'])
# # def qa():
# #     try:
# #         data = request.json
# #         question = data['question']
        
# #         # Check if the question is mathematical
# #         if is_mathematical_question(question):
# #             answer = solve_math_question(question)
# #             return jsonify({'answer': answer})
        
# #         # Fetch context from Google Custom Search for non-mathematical questions
# #         search_response = google_search(question, api_key, cse_id)
# #         if not search_response.get('items'):
# #             return jsonify({'answer': 'Could not find an answer on the web.'})

# #         # Aggregate snippets from multiple search results
# #         context = ""
# #         for item in search_response['items']:
# #             context += item['snippet'] + " "
# #             if len(context) > 1000:  # Limit context length to 1000 characters
# #                 break

# #         # Use the context with the QA pipeline
# #         result = qa_pipeline(question=question, context=context)
# #         print(f"QA Result: {result}")  # Debugging line
# #         return jsonify({'answer': result['answer']})
# #     except Exception as e:
# #         print(f"Error during question answering: {e}")  # Debugging line
# #         return jsonify({'error': str(e)}), 400

# # @app.route('/qa', methods=['POST'])
# # def qa():
# #     try:
# #         data = request.json
# #         question = data['question']

# #         # Fetch context from Google Custom Search
# #         search_response = google_search(question, api_key, cse_id)
# #         if not search_response.get('items'):
# #             return jsonify({'answer': 'Could not find an answer on the web.'})

# #         # Aggregate snippets from multiple search results
# #         context = ""
# #         for item in search_response['items']:
# #             context += item['snippet'] + " "
# #             if len(context) > 1000:  # Limit context length to 1000 characters
# #                 break

# #         # Use the context with the QA pipeline
# #         result = qa_pipeline(question=question, context=context)
# #         print(f"QA Result: {result}")  # Debugging line
# #         return jsonify({'answer': result['answer']})
# #     except Exception as e:
# #         print(f"Error during question answering: {e}")  # Debugging line
# #         return jsonify({'error': str(e)}), 400

# def is_mathematical_question(question):
#     # Simple heuristic to check if the question contains mathematical operators or numbers
#     math_keywords = ['+', '-', '*', '/', '=', 'sqrt', 'log', 'sin', 'cos', 'tan', '^', '(', ')']
#     if any(char.isdigit() for char in question) or any(op in question for op in math_keywords):
#         return True
#     return False

# def solve_math_question(question):
#     # Use an API to solve mathematical questions
#     api_url = f"http://api.mathjs.org/v4/?expr={requests.utils.quote(question)}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         return response.text
#     return "Could not solve the mathematical question."
# # @app.route('/qa', methods=['POST'])
# # def qa():
# #     try:
# #         data = request.json
# #         question = data['question']

# #         # Fetch context from Wikipedia
# #         search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={question}&format=json"
# #         headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyApp/0.1; +https://www.example.com/your-contact-info)'}
        
# #         search_response = requests.get(search_url, headers=headers)
# #         search_result = search_response.json()

# #         if not search_result['query']['search']:
# #             return jsonify({'answer': 'Could not find an answer on the web.'})

# #         page_title = search_result['query']['search'][0]['title']
# #         page_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&format=json&titles={page_title}"

# #         page_response = requests.get(page_url, headers=headers)
# #         page_result = page_response.json()

# #         page = next(iter(page_result['query']['pages'].values()))
# #         context = page.get('extract', '')

# #         if not context:
# #             return jsonify({'answer': 'Could not find an answer on the web.'})

# #         # Use the context with the QA pipeline
# #         result = qa_pipeline(question=question, context=context)
# #         print(f"QA Result: {result}")  # Debugging line
# #         return jsonify({'answer': result['answer']})
# #     except Exception as e:
# #         print(f"Error during question answering: {e}")  # Debugging line
# #         return jsonify({'error': str(e)}), 400

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for question, student_answer in answers.items():
#             inputs = bert_tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
#             outputs = bert_model(**inputs)
#             prediction = torch.argmax(outputs.logits, dim=-1).item()

#             correct_answer = correct_answers[subject][question]['answer']
#             similarity_check = correct_answers[subject][question]['similarity_check']
#             keywords = str(correct_answers[subject][question]['keywords']).split(',')
#             order = correct_answers[subject][question]['order']
#             marks = question_marks[subject][question]

#             if prediction == 0:  # Assuming 0 is the label for correct answers
#                 total_score += marks
#                 results[question] = {
#                     'order': order,
#                     'student_answer': student_answer,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': marks
#                 }
#             else:
#                 if similarity_check:
#                     sentences = [student_answer, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in student_answer for keyword in keywords):
#                         total_score += marks / 2  # Partial credit for similar answers
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': marks / 2
#                         }
#                     else:
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[question] = {
#                         'order': order,
#                         'student_answer': student_answer,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)
#         print(f"Calculated grade: {grade}")  # Debugging line

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         print(f"Received prompt for generation: {data['prompt']}")  # Debugging line
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7, top_p=0.95)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print(f"Generated text: {generated}")  # Debugging line
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         print(f"Error during text generation: {e}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)






# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict
# import requests
# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# def calculate_grade(score, max_score):
#     percentage = (score / max_score) * 100
#     if percentage >= 90:
#         return 'A'
#     elif percentage >= 80:
#         return 'B'
#     elif percentage >= 70:
#         return 'C'
#     elif percentage >= 60:
#         return 'D'
#     else:
#         return 'F'

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for question, student_answer in answers.items():
#             inputs = tokenizer(student_answer, return_tensors="pt", padding=True, truncation=True)
#             outputs = model(**inputs)
#             prediction = torch.argmax(outputs.logits, dim=-1).item()

#             correct_answer = correct_answers[subject][question]['answer']
#             similarity_check = correct_answers[subject][question]['similarity_check']
#             keywords = str(correct_answers[subject][question]['keywords']).split(',')
#             order = correct_answers[subject][question]['order']
#             marks = question_marks[subject][question]

#             if prediction == 0:  # Assuming 0 is the label for correct answers
#                 total_score += marks
#                 results[question] = {
#                     'order': order,
#                     'student_answer': student_answer,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': marks
#                 }
#             else:
#                 if similarity_check:
#                     sentences = [student_answer, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in student_answer for keyword in keywords):
#                         total_score += marks / 2  # Partial credit for similar answers
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is partially correct.",
#                             'similarity': similarity_score,
#                             'marks': marks / 2
#                         }
#                     else:
#                         results[question] = {
#                             'order': order,
#                             'student_answer': student_answer,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[question] = {
#                         'order': order,
#                         'student_answer': student_answer,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         grade = calculate_grade(total_score, max_score)

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             # Ensure keywords is treated as a string
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         student_id = data.get('student_id')  # Assuming student_id is passed in the request
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         grade = calculate_grade(total_score, max_score)

#         # Send data to Spring Boot
#         result_data = {
#             'studentId': student_id,
#             'testName': subject,
#             'score': total_score,
#             'maxScore': max_score,
#             'grade': grade
#         }
#         response = requests.post('http://localhost:8080/api/results', json=result_data)  # Update with your Spring Boot service URL
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to save test result'}), 500

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score, 'grade': grade})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)
############################################
############################################
############################################
############################################
############################################
############################################
# # This code works.
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             # Ensure keywords is treated as a string
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']  # Include the order here

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)





#This code works, just does not order correctly
# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         # Reload CSV files after upload
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         # Reload CSV files after deletion
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)

# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             print(f"Questions for {subject}: {questions}")  # Debug statement
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName'].lower()  # Normalize to lowercase
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName').lower()  # Normalize to lowercase
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         else:
#             return jsonify({'message': 'Test not found in metadata.'}), 404

#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             print(f"Loading CSV file: {filename}")  # Debug statement
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     print(f"Loaded subjects: {subjects.keys()}")  # Debug statement
#     return subjects

# correct_answers = load_csv_files()
# print(f"Correct answers loaded: {correct_answers.keys()}")  # Debug statement
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         print(f"Fetching questions for subject: {subject}")  # Debug statement
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             return jsonify(questions), 200
#         else:
#             print(f"Subject not found: {subject}")  # Debug statement
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error fetching questions for {subject}: {str(e)}")  # Debug statement
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)



# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             if 'order' in df.columns:
#                 df = df.sort_values(by=['order'])
#                 subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             else:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject').lower()
#         answers = data.get('answers')
#         results = OrderedDict()
#         total_score = 0
#         max_score = sum(question_marks[subject].values())

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")

#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Load metadata
# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# # Save metadata
# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()  # Normalize to lowercase
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             if 'order' not in df.columns:
#                 raise KeyError(f"'order' column is missing in the CSV file: {filename}")
#             df = df.sort_values(by=['order'])  # Ensure questions are sorted by 'order'
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# # In classify_api.py
# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             # Ensure keywords is treated as a string
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')
#             order = correct_answers[subject][key]['order']

#             if value == correct_answer:
#                 results[key] = {
#                     'order': order,  # Include the order here
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'order': order,  # Include the order here
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'order': order,  # Include the order here
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'order': order,  # Include the order here
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# # @app.route('/evaluate', methods=['POST'])
# # def evaluate():
# #     try:
# #         data = request.json
# #         print(f"Received data: {data}")  # Debugging line
# #         subject = data.get('subject').lower()  # Normalize to lowercase
# #         answers = data.get('answers')
# #         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
# #         total_score = 0
# #         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

# #         for key, value in answers.items():
# #             correct_answer = correct_answers[subject][key]['answer']
# #             similarity_check = correct_answers[subject][key]['similarity_check']
# #             # Ensure keywords is treated as a string
# #             keywords = str(correct_answers[subject][key]['keywords']).split(',')

# #             if value == correct_answer:
# #                 results[key] = {
# #                     'student_answer': value,
# #                     'correct_answer': correct_answer,
# #                     'feedback': "Your answer is correct.",
# #                     'similarity': 1.0,
# #                     'marks': question_marks[subject][key]  # Include marks
# #                 }
# #                 total_score += question_marks[subject][key]
# #             else:
# #                 if similarity_check:
# #                     sentences = [value, correct_answer]
# #                     embeddings = sentence_model.encode(sentences)
# #                     cosine_sim = torch.nn.functional.cosine_similarity(
# #                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
# #                     )
# #                     similarity_score = cosine_sim.item()
# #                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
# #                         results[key] = {
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is correct.",
# #                             'similarity': similarity_score,
# #                             'marks': question_marks[subject][key]  # Include marks
# #                         }
# #                         total_score += question_marks[subject][key]
# #                     else:
# #                         results[key] = {
# #                             'student_answer': value,
# #                             'correct_answer': correct_answer,
# #                             'feedback': "Your answer is incorrect.",
# #                             'similarity': similarity_score,
# #                             'marks': 0  # No marks if incorrect
# #                         }
# #                 else:
# #                     results[key] = {
# #                         'student_answer': value,
# #                         'correct_answer': correct_answer,
# #                         'feedback': "Your answer is incorrect.",
# #                         'similarity': 0,
# #                         'marks': 0  # No marks if incorrect
# #                     }

# #         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
# #     except Exception as e:
# #         return jsonify({'error': str(e)}), 400


# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")
        
#         # Check if the file exists before attempting to delete it
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             message = 'Test deleted successfully.'
#         else:
#             message = 'File not found.'
        
#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         return jsonify({'message': message, 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()  # Normalize to lowercase
#         if subject in correct_answers:
#             questions = [{'order': correct_answers[subject][q]['order'], 'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             questions = sorted(questions, key=lambda x: x['order'])  # Sort by order
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400



# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Load metadata
# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# # Save metadata
# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0].lower()  # Normalize to lowercase
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject').lower()  # Normalize to lowercase
#         answers = data.get('answers')
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             # Ensure keywords is treated as a string
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")
        
#         # Check if the file exists before attempting to delete it
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404
        
#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         subject = subject.lower()  # Normalize to lowercase
#         if subject in correct_answers:
#             questions = [{'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)


# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime
# from collections import OrderedDict

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Load metadata
# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# # Save metadata
# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0]
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = OrderedDict()  # Use OrderedDict to preserve the order of questions
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             # Ensure keywords is treated as a string
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")
        
#         # Check if the file exists before attempting to delete it
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404
        
#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         if subject in correct_answers:
#             questions = [{'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)



# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = '/Users/shafiulchowdhury/ai-assessments/python-api/csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Load metadata
# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# # Save metadata
# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0]
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#             print(f"Loaded {subject_name} with questions: {list(subjects[subject_name].keys())}")  # Debugging line
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         print(f"Received data: {data}")  # Debugging line
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = {}
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             # Ensure keywords is treated as a string
#             keywords = str(correct_answers[subject][key]['keywords']).split(',')

#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0,
#                     'marks': question_marks[subject][key]  # Include marks
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score,
#                             'marks': question_marks[subject][key]  # Include marks
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score,
#                             'marks': 0  # No marks if incorrect
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0,
#                         'marks': 0  # No marks if incorrect
#                     }

#         print(f"Results: {results}")  # Debugging line
#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         print(f"Error: {str(e)}")  # Debugging line
#         return jsonify({'error': str(e)}), 400


# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         file_path = os.path.join(csv_directory, f"{test_name}.csv")
        
#         # Check if the file exists before attempting to delete it
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         else:
#             return jsonify({'message': 'File not found.'}), 404
        
#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)
#         return jsonify({'message': 'Test deleted successfully.', 'files': list_files_in_directory()}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         print(f"Request for questions of subject: {subject}")  # Debugging line
#         if subject in correct_answers:
#             questions = [{'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             print(f"Returning questions for {subject}: {questions}")  # Debugging line
#             return jsonify(questions), 200
#         else:
#             print(f"Subject {subject} not found")  # Debugging line
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         print(f"Error retrieving questions for {subject}: {str(e)}")  # Debugging line
#         return jsonify({'error': str(e)}), 400

# @app.route('/list-files', methods=['GET'])
# def list_files():
#     try:
#         files = list_files_in_directory()
#         return jsonify({'files': files}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def list_files_in_directory():
#     try:
#         files = [f for f in os.listdir(csv_directory) if os.path.isfile(os.path.join(csv_directory, f))]
#         return files
#     except Exception as e:
#         return []

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)



# import os
# import pandas as pd
# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# from datetime import datetime

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# csv_directory = 'csv_files'
# metadata_file = 'csv_files_metadata.json'

# # Load metadata
# def load_metadata():
#     if os.path.exists(metadata_file):
#         with open(metadata_file, 'r') as file:
#             return json.load(file)
#     return {}

# # Save metadata
# def save_metadata(metadata):
#     with open(metadata_file, 'w') as file:
#         json.dump(metadata, file, indent=4)

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0]
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = {}
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = correct_answers[subject][key]['keywords'].split(',')
            
#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         timestamp = datetime.now().isoformat()
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         metadata[test_name] = timestamp
#         save_metadata(metadata)

#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         os.remove(os.path.join(csv_directory, f"{test_name}.csv"))

#         metadata = load_metadata()
#         if test_name in metadata:
#             del metadata[test_name]
#             save_metadata(metadata)

#         return jsonify({'message': 'Test deleted successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         metadata = load_metadata()
#         return jsonify(metadata), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         if subject in correct_answers:
#             questions = [{'question': q, 'name': q} for q in correct_answers[subject].keys()]
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     app.run(host='0.0.0.0', port=5001)



# import os
# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch
# import time

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# # Ensure CSV directory exists
# csv_directory = 'csv_files'
# if not os.path.exists(csv_directory):
#     os.makedirs(csv_directory)

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0]
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# # Load CSV files into memory
# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = {}
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = correct_answers[subject][key]['keywords'].split(',')
            
#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         os.remove(os.path.join(csv_directory, f"{test_name}.csv"))
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
#         return jsonify({'message': 'Test deleted successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         tests = []
#         for filename in os.listdir(csv_directory):
#             if filename.endswith('.csv'):
#                 filepath = os.path.join(csv_directory, filename)
#                 timestamp = os.path.getmtime(filepath)
#                 tests.append({
#                     'name': filename.split('.')[0],
#                     'timestamp': time.ctime(timestamp)
#                 })
#         return jsonify({'tests': tests}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         if subject in correct_answers:
#             questions = [{'question': q, 'name': q} for q in correct_answers[subject]]
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)


# import os
# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# # Ensure CSV directory exists
# csv_directory = 'csv_files'
# if not os.path.exists(csv_directory):
#     os.makedirs(csv_directory)

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0]
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# # Load CSV files into memory
# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = {}
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = correct_answers[subject][key]['keywords'].split(',')
            
#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         file.save(os.path.join(csv_directory, f"{test_name}.csv"))
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         os.remove(os.path.join(csv_directory, f"{test_name}.csv"))
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
#         return jsonify({'message': 'Test deleted successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         tests = list(correct_answers.keys())
#         return jsonify({'tests': tests}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         if subject in correct_answers:
#             questions = [{'question': q, 'name': q} for q in correct_answers[subject]]
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)




# import os
# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     csv_directory = 'csv_files'
#     if not os.path.exists(csv_directory):
#         os.makedirs(csv_directory)
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0]
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = {}
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = correct_answers[subject][key]['keywords'].split(',')
            
#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         file.save(os.path.join('csv_files', f"{test_name}.csv"))
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         os.remove(os.path.join('csv_files', f"{test_name}.csv"))
#         global correct_answers, question_marks
#         correct_answers = load_csv_files()
#         question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
#         return jsonify({'message': 'Test deleted successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-tests', methods=['GET'])
# def get_tests():
#     try:
#         tests = list(correct_answers.keys())
#         return jsonify({'tests': tests}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/get-questions/<subject>', methods=['GET'])
# def get_questions(subject):
#     try:
#         if subject in correct_answers:
#             questions = [{'question': q, 'name': q} for q in correct_answers[subject]]
#             return jsonify(questions), 200
#         else:
#             return jsonify({'error': 'Subject not found'}), 404
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)




# import os
# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# # Load the CSV files dynamically
# def load_csv_files():
#     subjects = {}
#     csv_directory = 'csv_files'
#     for filename in os.listdir(csv_directory):
#         if filename.endswith('.csv'):
#             subject_name = filename.split('.')[0]
#             df = pd.read_csv(os.path.join(csv_directory, filename))
#             subjects[subject_name] = df.set_index('question').to_dict(orient='index')
#     return subjects

# correct_answers = load_csv_files()
# question_marks = {subject: {q: data['marks'] for q, data in questions.items()} for subject, questions in correct_answers.items()}
# similarity_threshold = 0.65

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = {}
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject][key]['answer']
#             similarity_check = correct_answers[subject][key]['similarity_check']
#             keywords = correct_answers[subject][key]['keywords'].split(',')
            
#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 if similarity_check:
#                     sentences = [value, correct_answer]
#                     embeddings = sentence_model.encode(sentences)
#                     cosine_sim = torch.nn.functional.cosine_similarity(
#                         torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                     )
#                     similarity_score = cosine_sim.item()
#                     if similarity_score > similarity_threshold or any(keyword in value for keyword in keywords):
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is correct.",
#                             'similarity': similarity_score
#                         }
#                         total_score += question_marks[subject][key]
#                     else:
#                         results[key] = {
#                             'student_answer': value,
#                             'correct_answer': correct_answer,
#                             'feedback': "Your answer is incorrect.",
#                             'similarity': similarity_score
#                         }
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': 0
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/upload-csv', methods=['POST'])
# def upload_csv():
#     try:
#         file = request.files['file']
#         test_name = request.form['testName']
#         file.save(os.path.join('csv_files', f"{test_name}.csv"))
#         return jsonify({'message': 'CSV file uploaded successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/delete-test', methods=['POST'])
# def delete_test():
#     try:
#         test_name = request.json.get('testName')
#         os.remove(os.path.join('csv_files', f"{test_name}.csv"))
#         return jsonify({'message': 'Test deleted successfully.'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")
# sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# # Predefined correct answers for simplicity
# correct_answers = {
#     'geography': {
#         "q1": "A city with a population of minimum 5 million.",
#         "q2": "Air that is descending is compressed. This causes it to warm up and does not allow condensation to not take place."
#     },
#     'computer-science': {
#         "q1": "The CPU handles tasks such as fetch, decode, and execute instructions.",
#         "q2": "Malware is software designed to cause harm to computer systems."
#     },
#     'science': {
#         "q1": "Earth",
#         "q2": "Cholesterol"
#     },
#     'math': {
#         "q1": "4",
#         "q2": "4"
#     },
#     'history': {
#         "q1": "2 million",
#         "q2": "Blockade means an effort to cut off food, supplies, war material, or communications from a particular area by force, usually by naval power."
#     }
# }

# question_marks = {
#     'geography': {
#         "q1": 1,
#         "q2": 2
#     },
#     'computer-science': {
#         "q1": 1,
#         "q2": 1
#     },
#     'science': {
#         "q1": 1,
#         "q2": 1
#     },
#     'math': {
#         "q1": 1,
#         "q2": 1
#     },
#     'history': {
#         "q1": 1,
#         "q2": 1
#     }
# }

# # Define required keywords for each answer
# keywords = {
#     'history': {
#         "q1": ["2", "million"],
#         "q2": ["Blockade", "cut off", "food", "supplies", "war material", "communications", "force", "naval power"]
#     },
#     'geography': {
#         "q1": ["city", "population", "5 million"],
#         "q2": ["descending", "compressed", "warm up", "condensation"]
#     },
#     'computer-science': {
#         "q1": ["CPU", "fetch", "decode", "execute"],
#         "q2": ["Malware", "software", "harm", "computer systems"]
#     },
#     'science': {
#         "q1": ["Earth"],
#         "q2": ["Cholesterol"]
#     },
#     'math': {
#         "q1": ["4"],
#         "q2": ["4"]
#     }
# }

# similarity_threshold = 0.65  # Adjusted threshold for considering an answer correct based on similarity

# @app.route('/classify', methods=['POST'])
# def classify():
#     try:
#         data = request.json
#         inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#         outputs = model(**inputs)
#         predictions = torch.argmax(outputs.logits, dim=-1)
#         return jsonify({'classification': predictions.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# def evaluate_keywords(student_answer, correct_keywords):
#     return all(keyword in student_answer for keyword in correct_keywords)

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     try:
#         data = request.json
#         subject = data.get('subject')
#         answers = data.get('answers')
#         results = {}
#         total_score = 0
#         max_score = sum(question_marks[subject].values())  # Total possible score for the subject

#         for key, value in answers.items():
#             correct_answer = correct_answers[subject].get(key)
#             correct_keywords = keywords[subject].get(key)
#             if value == correct_answer:
#                 results[key] = {
#                     'student_answer': value,
#                     'correct_answer': correct_answer,
#                     'feedback': "Your answer is correct.",
#                     'similarity': 1.0
#                 }
#                 total_score += question_marks[subject][key]
#             else:
#                 sentences = [value, correct_answer]
#                 embeddings = sentence_model.encode(sentences)
#                 cosine_sim = torch.nn.functional.cosine_similarity(
#                     torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#                 )
#                 similarity_score = cosine_sim.item()
#                 if similarity_score > similarity_threshold or evaluate_keywords(value, correct_keywords):
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is correct.",
#                         'similarity': similarity_score
#                     }
#                     total_score += question_marks[subject][key]
#                 else:
#                     results[key] = {
#                         'student_answer': value,
#                         'correct_answer': correct_answer,
#                         'feedback': "Your answer is incorrect.",
#                         'similarity': similarity_score
#                     }

#         return jsonify({'results': results, 'score': total_score, 'max_score': max_score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     try:
#         data = request.json
#         sentences = [data['sentence1'], data['sentence2']]
#         embeddings = sentence_model.encode(sentences)
#         cosine_sim = torch.nn.functional.cosine_similarity(
#             torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0
#         )
#         return jsonify({'similarity': cosine_sim.item()})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/generate', methods=['POST'])
# def generate():
#     try:
#         data = request.json
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)












# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer
# from sentence_transformers import SentenceTransformer
# import torch

# app = Flask(__name__)
# CORS(app)

# # Load tokenizer and model
# bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# bert_model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")

# # Initialize Sentence-BERT
# sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')

# # Initialize T5
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# # Predefined correct answers for simplicity
# correct_answers = {
#     "q1": "4",
#     "q2": "4"
# }

# @app.route('/classify', methods=['POST'])
# def classify():
#     data = request.json
#     inputs = bert_tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#     outputs = bert_model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=-1)
#     return jsonify({'classification': predictions.item()})

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     data = request.json
#     results = {}
#     for key, value in data.items():
#         if value == correct_answers.get(key):
#             results[key] = f"The answer '{value}' is correct."
#         else:
#             results[key] = f"The answer '{value}' is incorrect."
#     return jsonify(results)

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     data = request.json
#     sentences = [data['sentence1'], data['sentence2']]
#     embeddings = sentence_model.encode(sentences)
#     cosine_sim = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[0]), torch.tensor(embeddings[1]), dim=0)
#     return jsonify({'similarity': cosine_sim.item()})

# # @app.route('/generate', methods=['POST'])
# # def generate():
# #     data = request.json
# #     inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
# #     outputs = t5_model.generate(inputs, max_length=50, num_return_sequences=1)
# #     generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
# #     return jsonify({'generated_text': generated})
# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.json
#     try:
#         inputs = t5_tokenizer.encode(data['prompt'], return_tensors='pt')
#         outputs = t5_model.generate(inputs, max_length=150, num_return_sequences=1)
#         generated = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return jsonify({'generated_text': generated})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)



# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, BertForSequenceClassification
# import torch

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")

# # Predefined correct answers for simplicity
# correct_answers = {
#     "q1": "4",
#     "q2": "4"
# }

# @app.route('/classify', methods=['POST'])
# def classify():
#     data = request.json
#     inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=-1)
#     return jsonify({'classification': predictions.item()})

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     data = request.json
#     results = {}
#     for key, value in data.items():
#         if value == correct_answers.get(key):
#             results[key] = f"The answer '{value}' is correct."
#         else:
#             results[key] = f"The answer '{value}' is incorrect."
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
# import torch
# import pandas as pd

# app = Flask(__name__)
# CORS(app)  # Enable CORS

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# @app.route('/classify', methods=['POST'])
# def classify():
#     data = request.json
#     inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=-1)
#     return jsonify({'classification': predictions.item()})

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     data = request.json
#     sentences = [data['sentence1'], data['sentence2']]
#     embeddings = model.encode(sentences)
#     cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
#     return jsonify({'similarity': cosine_sim.item()})

# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.json
#     inputs = tokenizer.encode(data['prompt'], return_tensors='pt')
#     outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
#     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({'generated_text': generated})

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     data = request.json
#     test_name = data['testName']
#     answers = data['answers']
#     # Example logic to process the answers and return results
#     # You can implement your logic here to evaluate the answers
#     # For demonstration, let's just return the received answers
#     results = {q_id: f"Processed answer for {answer}" for q_id, answer in answers.items()}
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)

# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification
# import torch
# import pandas as pd

# app = Flask(__name__)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# @app.route('/classify', methods=['POST'])
# def classify():
#     data = request.json
#     inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#     outputs = model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=-1)
#     return jsonify({'classification': predictions.item()})

# @app.route('/similarity', methods=['POST'])
# def similarity():
#     data = request.json
#     sentences = [data['sentence1'], data['sentence2']]
#     embeddings = model.encode(sentences)
#     cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
#     return jsonify({'similarity': cosine_sim.item()})

# @app.route('/generate', methods=['POST'])
# def generate():
#     data = request.json
#     inputs = tokenizer.encode(data['prompt'], return_tensors='pt')
#     outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
#     generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return jsonify({'generated_text': generated})

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     data = request.json
#     test_name = data['testName']
#     answers = data['answers']
#     # Example logic to process the answers and return results
#     # You can implement your logic here to evaluate the answers
#     # For demonstration, let's just return the received answers
#     results = {q_id: f"Processed answer for {answer}" for q_id, answer in answers.items()}
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# from flask import Flask, request, jsonify
# from transformers import BertTokenizer, BertForSequenceClassification
# import torch

# app = Flask(__name__)

# # Load tokenizer and fine-tuned model
# tokenizer = BertTokenizer.from_pretrained("./fine_tuned_model")
# model = BertForSequenceClassification.from_pretrained("./fine_tuned_model")

# @app.route('/classify', methods=['POST'])
# def classify():
#     data = request.json
#     print("Received data:", data)  # Debug statement
#     inputs = tokenizer(data['text'], return_tensors="pt", padding=True, truncation=True)
#     print("Tokenized inputs:", inputs)  # Debug statement
#     outputs = model(**inputs)
#     print("Model outputs:", outputs)  # Debug statement
#     predictions = torch.argmax(outputs.logits, dim=-1)
#     print("Predictions:", predictions)  # Debug statement
#     return jsonify({'classification': predictions.item()})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# import pandas as pd
# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# app = Flask(__name__)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# # Load data
# math_data = pd.read_csv('math_data.csv')
# science_data = pd.read_csv('science_data.csv')
# history_data = pd.read_csv('history_data.csv')
# geography_data = pd.read_csv('geography_data.csv')
# computerscience_data = pd.read_csv('computerscience_data.csv')

# # Combine all data into a single DataFrame
# data = pd.concat([math_data, science_data, history_data, geography_data, computerscience_data])

# def evaluate_answer(student_answer, correct_answer, marking_guidance):
#     score = 0
#     total_marks = 0

#     # Split answers into tokens or keywords
#     correct_tokens = correct_answer.split()
#     student_tokens = student_answer.split()

#     # Simple keyword matching
#     for token in correct_tokens:
#         if token in student_tokens:
#             score += 1

#     # Use marking guidance
#     if marking_guidance:
#         # Parse and apply marking guidance logic
#         parts = marking_guidance.split('.')
#         for part in parts:
#             if part:
#                 total_marks += 1
#                 if all(token in student_tokens for token in part.split()):
#                     score += 1

#     return min(score, total_marks)

# @app.route('/evaluate', methods=['POST'])
# def evaluate():
#     data = request.json
#     question = data['question']
#     student_answer = data['answer']

#     # Find the relevant question in the dataset
#     relevant_data = data[data['question'] == question].iloc[0]
#     correct_answer = relevant_data['answer']
#     marking_guidance = relevant_data['marking_guidance']

#     # Evaluate the answer
#     score = evaluate_answer(student_answer, correct_answer, marking_guidance)
    
#     return jsonify({'score': score})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001)
