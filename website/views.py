from flask import Blueprint, render_template, request, redirect, url_for
from .models import chunker, gen_a, highlight_answer, gen_q, tokenizer

views = Blueprint("views", __name__)

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        text = request.form.get('text')
        chunked_text = chunker(text, tokenizer)

        questions = []
        for chunk in chunked_text:
            answer_text = gen_a(chunk)
            highlighted_text = highlight_answer(chunk, answer_text, tokenizer)
            question_text = gen_q(highlighted_text)
            questions.append(question_text)


        return render_template("home.html", questions=questions)
    
  a  return render_template('home.html')