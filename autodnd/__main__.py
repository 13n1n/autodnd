from waitress import serve

from .game import app

serve(app, port=8996)