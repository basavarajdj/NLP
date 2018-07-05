import os
from flask import Blueprint
from flask import Flask, render_template, flash, request, url_for, redirect, Response, send_file
from functions import credentials, model_run
from os import remove,path
#import json
import pandas as pd
import sys

dashboard = Blueprint('dashboard',__name__,)

@dashboard.route('/', methods=["GET","POST"])
def index():
    error = ''
    try:
        if request.method == "POST":
            attempted_username = request.form['username']
            attempted_password = request.form['password']
            credt=credentials()
            lst=credt[attempted_username]
            if attempted_username in credt and lst[0]==attempted_password:
                #print("sucess!!!!")
                return render_template("dashboard.html")
            else:
                error = "Invalid credentials. Try Again."
                print(error)
                flash(error)
        return render_template("main.html", error = error)
    except Exception as e:
        flash(e)
        print(e)
        return render_template("main.html", error = error)


@dashboard.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')