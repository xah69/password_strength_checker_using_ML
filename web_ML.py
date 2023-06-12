from flask import Flask, render_template, request
import pickle
import getpass

#tokenizer
def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character

# Load the model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the model from the file
with open('tdif.pkl', 'rb') as file:
    tdif = pickle.load(file)

#web interface
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/greet', methods=['GET', 'POST'])
def greet():
    if request.method == 'POST':
        user = request.form['user']
        data = tdif.transform([user]).toarray()
        output = model.predict(data)
        
        print(user)
        print(output)
        
        return f"PASSWORD is , {output[0]}!"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
