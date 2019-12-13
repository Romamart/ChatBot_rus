from flask import Flask, render_template, redirect, url_for, request
from RunBot import evaluateInput  # output chatbot sentence 

app = Flask(__name__)
messages = []
redirection = False


def generateMessage(fromUser):
    if fromUser:
        text = request.form['textToChatbot']
        user = "User"
    else:
        text = evaluateInput(messages[-1][1])
        user = "Chatbot"
    return [user, text]


@app.route('/', methods=['GET'])
def initial():
    return redirect(url_for('mainPage'))


@app.route('/main', methods=['GET'])
def mainPage():
    return render_template('index.html')


@app.route('/inference', methods=['GET'])
def inference():
    global redirection
    if redirection:
        messages.insert(0, generateMessage(False))
        redirection = False
    return render_template('inference.html', messages=messages)


@app.route('/reinforcment', methods=['GET'])
def reinforcment():
    return "Will appear later"

@app.route('/add_message', methods=['POST'])
def addMessages():
    messages.insert(0, generateMessage(True))
    global redirection
    redirection = True
    return redirect(url_for('inference'))


if __name__ == "__main__":
    app.run(debug=True)