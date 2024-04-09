from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
from .models import chunker, gen_a, highlight_answer, gen_q, tokenizer, evaluate_answer, generate_user_feedback
from werkzeug.utils import secure_filename
from .models import extract_text, extract_metadata, write_text_to_file, split_file, run_query, indexing_pipeline, apply_clean_context_to_dict
import os

views = Blueprint("views", __name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}


@views.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'GET':
        # reset session variables for a new session
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

    # run query
    output_dict = run_query([question])
    output_dict = apply_clean_context_to_dict(output_dict)

    # extract the first answer and its context
    if output_dict:

        answer, context = next(iter(output_dict.items()))
        # store context in session for quiz generation
        session['captured_texts'].append(context)
    else:
        answer, context = "Sorry, I couldn't find an answer.", ""

    session['questions_asked'] = session.get('questions_asked', 0) + 1
    # pass answer and context to template
    return render_template('home.html', answer=answer, context=context, allow_questions=True, show_start_quiz=session['questions_asked'] >= 3, in_quiz_phase=False)

@views.route('/start_quiz', methods=['GET'])
def start_quiz():
    session['in_quiz_phase'] = True
    if 'captured_texts' in session and session['captured_texts']:
        session['allow_questions'] = False
        # reset quiz index to start from the first question
        session['quiz_index'] = 0
        # redirect generate_quiz
        return redirect(url_for('views.generate_quiz'))
    else:
        flash("No context available for quiz generation. Please ask some questions first.")
        return redirect(url_for('views.upload_file'))


@views.route('/generate_quiz', methods=['GET', 'POST'])
def generate_quiz():
    # initialize session variables
    if 'quiz_index' not in session:
        session['quiz_index'] = 0
    if 'feedback' not in session:
        session['feedback'] = []

    # fetch current quiz session values
    quiz_index = session['quiz_index']
    contexts = session.get('captured_texts', [])
    feedback = session.get('feedback', [])

    # check for quiz completion
    if quiz_index >= len(contexts):
        session['in_quiz_phase'] = False  # Mark quiz as completed
        return render_template('quiz.html', quiz_completed=True,
                               completion_message="Quiz completed! Review your feedback below.",
                               feedback=feedback)

    # generate question and answer from current context
    current_context = contexts[quiz_index]
    chunked_text = chunker(current_context, tokenizer)
    answer = gen_a(chunked_text[0])[0] if isinstance(gen_a(chunked_text[0]), list) else gen_a(chunked_text[0])
    highlighted_text = highlight_answer(chunked_text[0], answer, tokenizer)
    question = gen_q(highlighted_text)[1] if isinstance(gen_q(highlighted_text), list) else gen_q(highlighted_text)

    # No Chunking method
    # tokenized_context = tokenizer.encode(current_context, add_special_tokens=False)
    # answer = gen_a(tokenized_context)[0] if isinstance(gen_a(tokenized_context), list) else gen_a(tokenized_context)
    # highlighted_text = highlight_answer(tokenized_context, answer, tokenizer)
    # question = gen_q(highlighted_text)[0] if isinstance(gen_q(highlighted_text), list) else gen_q(highlighted_text)

    # User answer submission
    if request.method == 'POST':
        user_answer = request.form.get('user_answer', '').strip()
        if user_answer:
            evaluation_result = evaluate_answer(current_context, question, answer, user_answer)
            feedback.append(generate_user_feedback(evaluation_result, current_context))  # append new feedback
            session['feedback'] = feedback  # update feedback in session
            quiz_index += 1  # move to next question
            session['quiz_index'] = quiz_index

    # prepare next question if available
    if quiz_index < len(contexts):
        next_context = contexts[quiz_index]
        next_chunked_text = chunker(next_context, tokenizer)
        next_question_answer = gen_a(next_chunked_text[0])[0] if isinstance(gen_a(next_chunked_text[0]), list) else gen_a(next_chunked_text[0])
        next_highlighted_text = highlight_answer(next_chunked_text[0], next_question_answer, tokenizer)
        next_question = gen_q(next_highlighted_text)[0] if isinstance(gen_q(next_highlighted_text), list) else gen_q(next_highlighted_text)
    else:
        next_question = None  # no more questions available

    # render template with current feedback and next question
    return render_template('quiz.html', feedback=feedback[-1] if feedback else None,
                           question=next_question, quiz_index=quiz_index)




