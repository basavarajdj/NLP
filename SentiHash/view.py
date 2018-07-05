import os
from flask import Blueprint
from flask import render_template, request, flash
from functions import credentials, model_run
import pandas as pd

view = Blueprint('view',__name__,)

@view.route('/', methods=['GET','POST'])
def index(): 
    error = ''
    try:
        if request.method == 'POST':
           txt=request.form['txt']
           file_names = model_run(txt,view.root_path)
           #print("File name is "+file_name)
           return render_template('view.html', titles = 'Sentiment Score for : '+txt, file_names = file_names)
        error = 'error getting request'
        return render_template('dashboard.html', error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template('dashboard.html', error = error)

@view.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')