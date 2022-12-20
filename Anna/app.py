from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///annaRental.sqlite3'
db = SQLAlchemy(app)


db = SQLAlchemy(app)
class users(db.Model):
   id = db.Column('user_id', db.Integer, primary_key = True)
   name = db.Column(db.String(100))
   email = db.Column(db.String(200))  
   addr = db.Column(db.String(200))
   pin = db.Column(db.String(10))

def __init__(self, name, city, addr,pin):
   self.name = name
   self.email = email
   self.addr = addr
   self.pin = pin

db.create_all()