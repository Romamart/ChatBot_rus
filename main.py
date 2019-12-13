from flask import Flask, render_template, redirect, url_for, request

import asyncio
import re

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
    

def handleListLinks(text):
    for link in reversed(re.findall(".*\n", text)):
         messages.insert(0, ["Chatbot", link[:-1]])


def generateMessage(fromUser):
    if fromUser:
        text = request.form['textToChatbot']
        user = "User"
        messages.insert(0, [user, text])
    else:
        user = "Chatbot"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        text = loop.run_until_complete(tcp_echo_client(messages[0][1], loop))
        loop.close()
        if text[:6] == "--list":
            handleListLinks(text[6:])
        else:
            messages.insert(0, [user, text])

        


@app.route('/', methods=['GET'])
def initial():
    return redirect(url_for('mainPage'))


@app.route('/main', methods=['GET'])
def mainPage():
    return render_template('index.html')


@app.route('/inference', methods=['GET'])
def inference():
    global redirection
    # Отмена вывода сбщ бота при первом заходе на стр inference
    if redirection:
        redirection = False
        generateMessage(redirection)
    return render_template('inference.html', messages=messages)


@app.route('/reinforcment', methods=['GET'])
def reinforcment():
    return "Will appear later"

@app.route('/add_message', methods=['POST'])
def addMessages():
    global redirection
    redirection = True
    generateMessage(redirection)
    return redirect(url_for('inference'))


if __name__ == "__main__":
    app.run(debug=True)