from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=['GET','POST'])
def predict():
    furnished=0;
    if request.method =="POST":
        houseType=request.form.get("type")
        houseSubType=request.form.get("sub_type")
        county=request.form.get("county")
        locality=request.form.get("locality")
        bedrooms=request.form.get("bedrooms")
        bathrooms=request.form.get("bathrooms")
        toilets=request.form.get("toilets")
        parking=request.form.get("toilets")
        furnished=request.form.get("furnished")
    return render_template('predict.html',prediction=furnished)

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