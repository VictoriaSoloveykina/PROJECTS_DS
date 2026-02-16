#export FLASK_APP=houseapp.py

from app import app

if __name__ == '__main__':
    app.run(debug=True)


