import os
from flask import Blueprint
from flask import send_file

download = Blueprint('download',__name__,)


@download.route('/')
def index():
    try:
        file_path=os.path.join(download.root_path,'final.csv')
        return send_file(file_path, attachment_filename='twitter.csv',as_attachment=True)
    except Exception as e:
        return str(e)