from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
from .models import chunker, gen_a, highlight_answer, gen_q, tokenizer, evaluate_answer, generate_user_feedback
from werkzeug.utils import secure_filename
from .models import extract_text, extract_metadata, write_text_to_file, split_file, run_query, indexing_pipeline
import os

views = Blueprint("views", __name__)

# Utility function moved here for accessibility
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}


@views.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'GET':
        # Reset session variables for a new session
        session['allow_questions'] = False
        session['captured_texts'] = []

    message = ''  # Message to display to the user

    if request.method == 'POST':
        if 'file' not in request.files:
            message = 'No file part'
        else:
            file = request.files['file']
            if file.filename == '':
                message = 'No selected file'
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                text = extract_text(file_path)
                write_text_to_file(text, "processed_text.txt")
                file_paths = split_file('processed_text.txt')
                paths = [path for path in file_paths]
                indexing_pipeline.run_batch(file_paths=paths)
                message = "we got it! let us take care of your material now!"
                session['allow_questions'] = True  # Enable question-asking
                session['captured_texts'] = []  # Initialize the list to store text pieces
    return render_template('home.html', message=message, allow_questions=session.get('allow_questions', False))


@views.route('/ask', methods=['POST'])
def ask_question():
    question = request.form.get('question')
    if not question:
        flash('Please enter a question.')
        return redirect(url_for('views.upload_file'))

    # Run the query
    output_dict = run_query([question])

    # Extract the first answer and its context
    if output_dict:
        # Get the first key-value pair from the dictionary
        answer, context = next(iter(output_dict.items()))
        # Store the context (relevant text) for quiz generation
        session['captured_texts'].append(context)
    else:
        answer, context = "Sorry, I couldn't find an answer.", ""

    session['questions_asked'] = session.get('questions_asked', 0) + 1
    # Pass both answer and context to the template
    return render_template('home.html', answer=answer, context=context, allow_questions=True, show_start_quiz=session['questions_asked'] >= 3, in_quiz_phase=False)

@views.route('/start_quiz', methods=['GET'])
def start_quiz():
    session['in_quiz_phase'] = True
    # Ensure there are captured contexts to generate a quiz from
    if 'captured_texts' in session and session['captured_texts']:
        session['allow_questions'] = False
        # Redirect to a new route that will handle quiz generation and display
        return redirect(url_for('views.generate_quiz'))
    else:
        # If no contexts are captured, inform the user and redirect back
        flash("No context available for quiz generation. Please ask some questions first.")
        return redirect(url_for('views.upload_file'))

@views.route('/generate_quiz', methods=['GET', 'POST'])
def generate_quiz():
    print("generate_quiz accessed")
    if not session.get('in_quiz_phase', False):
        print("Not in quiz phase, redirecting...")
        return redirect(url_for('views.upload_file'))

    if 'quiz_index' not in session:
        session['quiz_index'] = 0  # Initialize the quiz question index
        session['feedback'] = []  # Initialize feedback storage

    # Get the stored contexts and the current quiz index
    contexts = session.get('captured_texts', [])
    current_index = session['quiz_index']

    if current_index >= len(contexts):
        # All contexts have been processed, quiz is complete
        session['in_quiz_phase'] = False  # Exit quiz phase
        return render_template('home.html', completion_message="Quiz completed! Review your feedback below.",
                               feedback=session['feedback'], quiz_completed=True)

    # Process the current context
    current_context = contexts[current_index]
    chunked_text = chunker(current_context, tokenizer)
    questions, answers, feedback = [], [], []

    for chunk in chunked_text:
        # Generate question-answer pairs
        answer_text = gen_a(chunk)[0] if isinstance(gen_a(chunk), list) else gen_a(chunk)
        question_text = gen_q(chunk)[0] if isinstance(gen_q(chunk), list) else gen_q(chunk)
        questions.append(question_text)
        answers.append(answer_text)

    # Handle user's answer submission and feedback generation
    if request.method == 'POST':
        for idx, (question, answer) in enumerate(zip(questions, answers)):
            user_answer = request.form.get(f'user_answer_{idx}', '').strip()
            if user_answer:
                # Evaluate the user's answer
                evaluation_result = evaluate_answer(chunk, question, answer, user_answer)
                # Generate feedback based on the evaluation
                feedback.append(generate_user_feedback(evaluation_result, chunk))

    # Prepare for the next chunk or complete the quiz
    if request.method == 'POST' or len(questions) == 0:
        session['quiz_index'] += 1  # Move to the next context
        session['feedback'].extend(feedback)  # Store the generated feedback

    return render_template('home.html', questions=questions, feedback=feedback, in_quiz_phase=True)


# @views.route('/generate_quiz', methods=['GET', 'POST'])
# def generate_quiz():
#     if not session.get('in_quiz_phase', False):
#         return redirect(url_for('views.upload_file'))
#     # Ensure there are stored contexts to generate questions from
#     contexts = session.get('captured_texts', [])
#
#     # Initialize session variables if not present
#     if 'quiz_index' not in session:
#         session['quiz_index'] = 0
#         session['feedback'] = []
#
#     # If all questions have been answered, show quiz completion
#     if session['quiz_index'] >= len(contexts):
#         completion_message = "Quiz completed! Review your feedback below."
#         return render_template('home.html', completion_message=completion_message, feedback=session['feedback'],
#                                quiz_completed=True)
#
#     current_context = contexts[session['quiz_index']]
#     feedback = []
#
#     if request.method == 'POST':
#         user_answer = request.form.get('user_answer', '').strip()
#
#         # Generate question-answer pair from the current context
#         chunked_context = chunker(current_context, tokenizer)
#         # Ensure lists are converted to strings by taking the first element
#         answer_text = gen_a(chunked_context)[0] if isinstance(gen_a(chunked_context), list) else gen_a(chunked_context)
#         question_text = gen_q(chunked_context)[0] if isinstance(gen_q(chunked_context), list) else gen_q(
#             chunked_context)
#
#         if user_answer:
#             # Evaluate the user's answer and generate feedback
#             evaluation_result = evaluate_answer(current_context, question_text, answer_text, user_answer)
#             tailored_feedback = generate_user_feedback(evaluation_result, current_context)
#             feedback.append(tailored_feedback)
#             session['feedback'].append(tailored_feedback)
#
#             # Prepare the next question
#             session['quiz_index'] += 1
#
#     # Render the current question and any feedback
#     return render_template('home.html', question=question_text, feedback=feedback, show_next_button=True)


