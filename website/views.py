from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
from .models import chunker, gen_a, highlight_answer, gen_q, tokenizer, evaluate_answer, generate_user_feedback
from werkzeug.utils import secure_filename
from .models import extract_text, extract_metadata, write_text_to_file, split_file, run_query, indexing_pipeline, apply_clean_context_to_dict
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
    output_dict = apply_clean_context_to_dict(output_dict)

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
    if 'captured_texts' in session and session['captured_texts']:
        session['allow_questions'] = False
        # Initialize or reset the quiz index to start from the first question
        session['quiz_index'] = 0
        # Redirect to the generate_quiz route to start the quiz
        return redirect(url_for('views.generate_quiz'))
    else:
        # If no contexts are captured, inform the user and redirect back
        flash("No context available for quiz generation. Please ask some questions first.")
        return redirect(url_for('views.upload_file'))


@views.route('/generate_quiz', methods=['GET', 'POST'])
def generate_quiz():
    # Ensure 'quiz_index' and 'feedback' are initialized in session
    if 'quiz_index' not in session:
        session['quiz_index'] = 0
    if 'feedback' not in session:
        session['feedback'] = []

    # Fetch current quiz index, contexts, and feedback from session
    quiz_index = session['quiz_index']
    contexts = session.get('captured_texts', [])
    feedback = session.get('feedback', [])

    # Check for quiz completion
    if quiz_index >= len(contexts):
        session['in_quiz_phase'] = False  # Mark quiz as completed
        return render_template('quiz.html', quiz_completed=True,
                               completion_message="Quiz completed! Review your feedback below.",
                               feedback=feedback)

    # Get current context and generate question and answer
    current_context = contexts[quiz_index]
    tokenized_context = tokenizer.encode(current_context, add_special_tokens=False)
    question = gen_q(tokenized_context)[0] if isinstance(gen_q(tokenized_context), list) else gen_q(tokenized_context)
    answer = gen_a(tokenized_context)[0] if isinstance(gen_a(tokenized_context), list) else gen_a(tokenized_context)

    # Process user's answer submission
    if request.method == 'POST':
        user_answer = request.form.get('user_answer', '').strip()
        if user_answer:
            evaluation_result = evaluate_answer(current_context, question, answer, user_answer)
            feedback.append(generate_user_feedback(evaluation_result, current_context))  # Append new feedback
            session['feedback'] = feedback  # Update feedback in session
            quiz_index += 1  # Increment quiz index to move to next question
            session['quiz_index'] = quiz_index  # Update quiz index in session

    # Prepare next question if available
    if quiz_index < len(contexts):
        next_context = contexts[quiz_index]
        next_chunked_text = chunker(next_context, tokenizer)
        next_question = gen_q(next_chunked_text[0])[0] if isinstance(gen_q(next_chunked_text[0]), list) else gen_q(next_chunked_text[0])
    else:
        next_question = None  # No more questions available

    # Render template with current feedback and next question
    return render_template('quiz.html', feedback=feedback[-1] if feedback else None,
                           question=next_question, quiz_index=quiz_index)



## this code was the easier set up but it currently returns error. Kept if for reference regarding question answer handling
# @views.route('/generate_quiz', methods=['GET', 'POST'])
# def generate_quiz():
#     # Ensure 'quiz_index' and 'feedback' are initialized in session
#     if 'quiz_index' not in session:
#         session['quiz_index'] = 0
#     if 'feedback' not in session:
#         session['feedback'] = []
#
#     # Fetch current quiz index, contexts, and feedback from session
#     quiz_index = session['quiz_index']
#     contexts = session.get('captured_texts', [])
#     feedback = session.get('feedback', [])
#
#     current_context = contexts[quiz_index]
#     chunked_text = chunker(current_context, tokenizer)
#
#     if chunked_text:
#         question = gen_q(chunked_text[0])[0] if isinstance(gen_q(chunked_text[0]), list) else gen_q(
#             chunked_text[0])  # Use the first chunk for simplicity
#         answer = gen_a(chunked_text[0])[0] if isinstance(gen_a(chunked_text[0]), list) else gen_a(
#             chunked_text[0])  # Get the corresponding answer
#
#         if request.method == 'POST':
#             user_answer = request.form.get('user_answer', '').strip()
#             if user_answer:
#                 evaluation_result = evaluate_answer(current_context, question, answer, user_answer)
#                 feedback.append(generate_user_feedback(evaluation_result, current_context))
#                 session['feedback'] = feedback  # Update the session variable
#
#             quiz_index += 1  # Increment quiz index to move to next question
#             session['quiz_index'] = quiz_index
#
#             # Prepare next question if available
#             if quiz_index < len(contexts):
#                 next_context = contexts[quiz_index]
#                 next_chunked_text = chunker(next_context, tokenizer)
#                 next_question = gen_q(next_chunked_text[0])[0] if isinstance(gen_q(next_chunked_text[0]),
#                                                                              list) else gen_q(next_chunked_text[0])
#             else:
#                 next_question = None  # No more questions available
#
#             # Render template with current feedback and next question
#             return render_template('quiz.html', feedback=feedback[-1] if feedback else None,
#                                    question=next_question, quiz_index=quiz_index)
#     else:
#         # No chunks to generate questions from, consider quiz completed
#         session['in_quiz_phase'] = False
#         return render_template('quiz.html', quiz_completed=True,
#                                completion_message="Quiz completed! No more questions available.",
#                                feedback=session['feedback'])
#




