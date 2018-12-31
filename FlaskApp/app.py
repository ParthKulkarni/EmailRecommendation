from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
import model

app = Flask(__name__)

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'flaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# init MYSQL
mysql = MySQL(app)

# Index


@app.route('/')
def index():
    return render_template('home.html')


# About
@app.route('/about')
def about():
    return render_template('about.html')


#Single Thread
@app.route('/view_thread/<string:id>/')
def view_thread(id):
    # Create cursor
    cur = mysql.connection.cursor()

    # Get article
    result = cur.execute("SELECT * FROM mails WHERE thread_no = %s", [id])

    threads = cur.fetchall()

    if result > 0:
        return render_template('thread.html', threads=threads)
    else:
        msg = 'No mails Found'
        return render_template('thread.html', msg=msg)


# Register Form Class
class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        cur.execute("INSERT INTO users1(name, email, username, password) VALUES(%s, %s, %s, %s)",
                    (name, email, username, password))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('You are now registered and can log in', 'success')

        return redirect(url_for('index'))
    return render_template('register.html', form=form)


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute(
            "SELECT * FROM users1 WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username

                flash('You are now logged in', 'success')
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')


# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap


# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


def list_to_string(l) :
    s = ''
    for item in l :
        s = s + "'" + str(item) + "',"
    s = s[:-1]
    return s

# Dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
@is_logged_in
def dashboard():
    cur = mysql.connection.cursor()
    if request.method == 'POST':
        # Get Form Fields
        user = request.form['query_name']

        md = model.model()
        li = md.generate_random_list()
        str_li = list_to_string(li)
        
        result = cur.execute("SELECT * FROM threads WHERE id in ({first})".format(first=str_li))

        threads = cur.fetchall()

        if result > 0 :
            return render_template('dashboard.html', threads=threads)
        else :
            msg = 'No Articles Found'
            return render_template('dashboard.html', msg=msg)
        
        # Close connection
        # cur.close()
    # return render_template('dashboard.html')

    # # Create cursor
    # cur = mysql.connection.cursor()

    # # Get articles
    result1 = cur.execute("SELECT * FROM threads;")
    # # Show articles only from the user logged in
    # # result = cur.execute("SELECT * FROM threads WHERE author = %s", [session['username']])

    threads1 = cur.fetchall()

    if result1 > 0:  
        return render_template('dashboard.html', threads=threads1)
    else:
        msg = 'No Articles Found'
        return render_template('dashboard.html', msg=msg)
    # Close connection
    cur.close()


if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True)
