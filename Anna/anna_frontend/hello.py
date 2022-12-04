from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')
@app.route("/predict")
def predict():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/login")
def login():
    return render_template('authentication/login.html')

@app.route("/register")
def register():
    return render_template('authentication/register.html')

# handle 404 error
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
if __name__ == "__main__":
    app.run(debug=True)