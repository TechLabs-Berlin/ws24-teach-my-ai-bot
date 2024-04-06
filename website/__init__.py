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

    UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
    ALLOWED_EXTENSIONS = {'pdf'}  # Allowed file types for upload
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config.SESSION_COOKIE_SAMESITE = 'None'
    app.config.SESSION_COOKIE_SECURE = 'True'

    return app

    return app

