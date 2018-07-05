from flask import Flask, render_template, request
from dashboard  import dashboard
from view import view
from download import download

server = Flask(__name__)
server.register_blueprint(dashboard, url_prefix='/dashboard/')
server.register_blueprint(view, url_prefix='/view/')
server.register_blueprint(download, url_prefix='/download/')

server.secret_key = 'super secret key'


@server.route('/')
def index():
   return render_template('index.html')


if __name__ == '__main__':
    server.run(debug = True)