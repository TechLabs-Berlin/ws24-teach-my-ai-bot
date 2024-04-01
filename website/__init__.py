from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

def create_app():
    app.config["SECRET_KEY"] = "sonia"

    from .views import views
    from .auth import auth

    app.register_blueprint(views, url_prefix="/")
    app.register_blueprint(auth, url_prefix="/")
    # Configuration
    UPLOAD_FOLDER = 'uploads'  # Folder where uploaded files will be stored
    ALLOWED_EXTENSIONS = {'pdf'}  # Allowed file types for upload
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
    return app
