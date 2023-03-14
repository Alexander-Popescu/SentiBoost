from flask import Flask, request, render_template, redirect, url_for
from pickle import FALSE
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import math
from statistics import LinearRegression
import numpy as np
import pandas as pd
import statsmodels.api as sm

#import ml stuff ^
max_length = 512

global_input_text = ''

model = BertForSequenceClassification.from_pretrained("fake-news-bert-base-uncased", num_labels=2)
tokenizer = BertTokenizerFast.from_pretrained("fake-news-bert-base-uncased", do_lower_case=True)

def get_prediction(text, convert_to_label=False):
    # prepare text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    # perform inference to model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
        0: "reliable",
        1: "unreliable"
    }
    if convert_to_label:
      return d[int(probs.argmax())]
    else:
      return int(probs.argmax())

app = Flask(__name__)
#load model into model var

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        text = "error"
        text = request.form["specialtext"]
        return redirect(url_for("results", answer=text))
    else:
        return render_template('index.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    #return render_template('results.html')
    #To do: calculate all plots and error check them before rendering results
    if request.method == "POST":
        text = "error"
        text = request.form["specialtext"]
        global global_input_text
        global_input_text = text
        fakenewstrueorfalse = get_prediction(text)
        if fakenewstrueorfalse == 1:
            fakenews = "unreliable"
            bannercolor = 'redbanner.html'
        else:
            fakenews = "reliable"
            bannercolor = 'greenbanner.html'
        return render_template("results.html", answer=text, fakenews=fakenews, bannercolor=bannercolor)
    else:
        return render_template('index.html')

@app.route('/plot1.png')
def plot1_png():
    #create plot
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    #vertical red line

    #get input text to run the custom function on
    global global_input_text
    text = global_input_text

    xval = text.count('"') + text.count("'")

    xs = [xval,xval]
    ys = [0,1]
    axis.plot(xs, ys, "r-")

    #regression and point plot from csv
    data = pd.read_csv('quotenumber_cleaned.csv')

    x_data = data['Value']
    y_data = data['Label']

    #plot logistic regression curve

    log_reg = sm.Logit(y_data, x_data).fit()
    axis.scatter(x_data,y_data)

    y_pred = []
    x_pred = []

    #creates array from logistic regression predictions to plot the curve
    for i in np.linspace(0,250,250):
        y_pred.append(log_reg.predict(i))   
        x_pred.append(i)

    #plots the regression curves
    axis.plot(x_pred, y_pred, color='orange')

    # p = sns.regplot(x=x_data, y=y_data, data=data, logistic=True, ci=None)
    # x_reg = p.get_lines()[0].get_xdata()
    # y_reg = p.get_lines()[0].get_ydata()

    #plt.savefig(csv_name + '.png')
    #axis.title('count of ' + ' in article')
    # print(x_reg)
    # print(y_reg)

    #creates cool label thing for all things plotted
    axis.legend(['Input Text','Dataset Points', 'Probability Curve'])

    #forces output image to have the window at specified zoom, so the limits are the limits of what the viewport will show
    
    #also increase size to always fit red line on the screen
    xlimval = 126
    if xval > 126:
        xlimval = xval + 6 #6 is extra for cosmetic reasons
    
    axis.set(xlim=(-1,xlimval), ylim=(-0.1, 1.1), autoscale_on=False,
           title='Number of Quotes Used')

    #send the plot over to the html side of things
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot2.png')
def plot2_png():
    #create plot
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    #vertical red line

    #get input text to run the custom function on
    global global_input_text
    text = global_input_text

    words = len(global_input_text.split())
    sentences = global_input_text.count('.') + global_input_text.count('!') + global_input_text.count('?')
    if sentences == 0:
        sentences = 1
    xval = words / sentences

    xs = [xval,xval]
    ys = [0,1]
    axis.plot(xs, ys, "r-")

    #regression and point plot from csv
    data = pd.read_csv('words_per_sentence_cleaned.csv')

    x_data = data['Value']
    y_data = data['Label']

    #plot logistic regression curve

    log_reg = sm.Logit(y_data, x_data).fit()
    axis.scatter(x_data,y_data)

    y_pred = []
    x_pred = []

    #creates array from logistic regression predictions to plot the curve
    for i in np.linspace(0,250,250):
        y_pred.append(log_reg.predict(i))   
        x_pred.append(i)

    #plots the regression curves
    axis.plot(x_pred, y_pred, color='orange')

    # p = sns.regplot(x=x_data, y=y_data, data=data, logistic=True, ci=None)
    # x_reg = p.get_lines()[0].get_xdata()
    # y_reg = p.get_lines()[0].get_ydata()

    #plt.savefig(csv_name + '.png')
    #axis.title('count of ' + ' in article')
    # print(x_reg)
    # print(y_reg)

    #creates cool label thing for all things plotted
    axis.legend(['Input Text','Dataset Points', 'Probability Curve'])

    #forces output image to have the window at specified zoom, so the limits are the limits of what the viewport will show
    #also increase size to always fit red line on the screen
    xlimval = 160
    if xval > 160:
        xlimval = xval + 6 #6 is extra for cosmetic reasons
    
    axis.set(xlim=(-1,xlimval), ylim=(-0.1, 1.1), autoscale_on=False,
           title='Word Density in Average Sentence')

    #send the plot over to the html side of things
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot3.png')
def plot3_png():
    #create plot
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    #vertical red line

    #get input text to run the custom function on
    global global_input_text
    text = global_input_text

    letters = 0
    words = [0]
    for i in range(0, len(text)):
        if text[i] != ' ':
            letters = letters + 1
        else:
            words.append(letters)
            letters = 0
    addition = 0
    for i in range(0, len(words)):
        addition = addition + words[i]
    xval = addition / len(words)

    xs = [xval,xval]
    ys = [0,1]
    axis.plot(xs, ys, "r-")

    #regression and point plot from csv
    data = pd.read_csv('average_letter_per_word_cleaned.csv')

    x_data = data['Value']
    y_data = data['Label']

    #plot logistic regression curve

    log_reg = sm.Logit(y_data, x_data).fit()
    axis.scatter(x_data,y_data)

    y_pred = []
    x_pred = []

    #creates array from logistic regression predictions to plot the curve
    for i in np.linspace(0,250,250):
        y_pred.append(log_reg.predict(i))   
        x_pred.append(i)

    #plots the regression curves
    axis.plot(x_pred, y_pred, color='orange')

    # p = sns.regplot(x=x_data, y=y_data, data=data, logistic=True, ci=None)
    # x_reg = p.get_lines()[0].get_xdata()
    # y_reg = p.get_lines()[0].get_ydata()

    #plt.savefig(csv_name + '.png')
    #axis.title('count of ' + ' in article')
    # print(x_reg)
    # print(y_reg)

    #creates cool label thing for all things plotted
    axis.legend(['Input Text','Dataset Points', 'Probability Curve'])

    #forces output image to have the window at specified zoom, so the limits are the limits of what the viewport will show
    #also increase size to always fit red line on the screen
    xlimval = 160
    if xval > 160:
        xlimval = xval + 6 #6 is extra for cosmetic reasons
    
    axis.set(xlim=(-1,xlimval), ylim=(-0.1, 1.1), autoscale_on=False,
           title='Average letters per word')

    #send the plot over to the html side of things
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plot4.png')
def plot4_png():
    #create plot
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    #vertical red line

    #get input text to run the custom function on
    global global_input_text
    text = global_input_text

    xval = text.count('.') + text.count(',')/ (text.count('!') + text.count('?') + 1)

    xs = [xval,xval]
    ys = [0,1]
    axis.plot(xs, ys, "r-")

    #regression and point plot from csv
    data = pd.read_csv('advanced_grammer_usage.csv')

    x_data = data['Value']
    y_data = data['Label']

    #plot logistic regression curve

    log_reg = sm.Logit(y_data, x_data).fit()
    axis.scatter(x_data,y_data)

    y_pred = []
    x_pred = []

    #creates array from logistic regression predictions to plot the curve
    for i in np.linspace(0,250,250):
        y_pred.append(log_reg.predict(i))   
        x_pred.append(i)

    #plots the regression curves
    axis.plot(x_pred, y_pred, color='orange')

    # p = sns.regplot(x=x_data, y=y_data, data=data, logistic=True, ci=None)
    # x_reg = p.get_lines()[0].get_xdata()
    # y_reg = p.get_lines()[0].get_ydata()

    #plt.savefig(csv_name + '.png')
    #axis.title('count of ' + ' in article')
    # print(x_reg)
    # print(y_reg)

    #creates cool label thing for all things plotted
    axis.legend(['Input Text','Dataset Points', 'Probability Curve'])

    #forces output image to have the window at specified zoom, so the limits are the limits of what the viewport will show
    #also increase size to always fit red line on the screen
    xlimval = 160
    if xval > 160:
        xlimval = xval + 6 #6 is extra for cosmetic reasons
    
    axis.set(xlim=(-1,xlimval), ylim=(-0.1, 1.1), autoscale_on=False,
           title='Advanced Grammatical Usage')

    #send the plot over to the html side of things
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == "__main__":
    app.run()