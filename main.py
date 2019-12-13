from flask import Flask, render_template, redirect, url_for, request

import asyncio

app = Flask(__name__)
messages = []
redirection = False


async def tcp_echo_client(message, loop):
    try:
        reader, writer = await asyncio.open_connection('127.0.0.1', 8888, loop=loop)
        writer.write(message.encode())
        data = await reader.read(100)
        writer.close()
        return data.decode()
    except:
        return "*еле уловимый шепот* Оживите меня."
    

def generateMessage(fromUser):
    if fromUser:
        text = request.form['textToChatbot']
        user = "User"
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        text = loop.run_until_complete(tcp_echo_client(messages[0][1], loop))
        loop.close()
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