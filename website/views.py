from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
# from .models import chunker, gen_a, highlight_answer, gen_q, tokenizer
from werkzeug.utils import secure_filename
from .models import extract_text, extract_metadata
import os

views = Blueprint("views", __name__)

# Utility function moved here for accessibility
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}


@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Your POST request handling logic here
        # For now, we're simply redirecting back to home for demonstration purposes
        return redirect(url_for('views.home'))

    # This return statement will handle GET requests and any other cases not covered by the above logic
    return render_template('home.html')

@views.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')  # Flash a message
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')  # Flash another message
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            flash('File successfully uploaded')  # Success message
            return redirect(url_for('views.upload_file'))  # Redirect back or to another page
    return render_template('home.html', uploaded=False)

