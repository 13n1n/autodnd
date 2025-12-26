from waitress import serve

from .api import app

serve(app, port=8996)