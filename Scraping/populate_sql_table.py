import MySQLdb, random, glob, pprint
import read_file
from dateutil.parser import parse

# Open database connection
db = MySQLdb.connect("localhost","root","1234","flaskapp")

# prepare a cursor object using cursor() method
cursor = db.cursor()

db.set_character_set('utf8')
cursor.execute('SET NAMES utf8;')
cursor.execute('SET CHARACTER SET utf8;')
cursor.execute('SET character_set_connection=utf8;')

sql = "CREATE TABLE threads(id INT(11) AUTO_INCREMENT PRIMARY KEY, author VARCHAR(100), subject VARCHAR(255), content TEXT, date TIMESTAMP);"
cursor.execute(sql)
sql = "CREATE TABLE mails(id INT(11) AUTO_INCREMENT PRIMARY KEY, thread_no INT(11),author VARCHAR(100), subject VARCHAR(255), content TEXT, date TIMESTAMP);"
cursor.execute(sql)
sql = "ALTER TABLE threads CONVERT TO CHARACTER SET utf8"
cursor.execute(sql)
sql = "ALTER TABLE mails CONVERT TO CHARACTER SET utf8"
cursor.execute(sql)

dir_path = '/home/vikas/Projects/debian/debian_dataset/'

for x in range (1, 100) :
    fpath_ = dir_path + str(x) + '/*.txt'
    files = glob.glob(fpath_)
    threads = []
    for file in files :
        cnt = 0
        ob = read_file.file_content(file)
        ob.read_file_content()
        threads.append(ob.mail)
    sorted_threads = sorted(threads, key=lambda k: k['Date'])
    for k in sorted_threads :
        # pprint.pprint(k['Date'])
        dt = str(parse(k['Date'])).split('+')[0]
        dt = dt[:19]
        if cnt == 0 :
            cursor.execute("""INSERT into threads (author, subject, content, date) values(%s,%s,%s,%s)""",(str(k['From'].split('<')[0][:-1]), str(k['Subject']), str(k['content']), str(dt)))
            db.commit()
        print("""INSERT into mails (thread_no, author, subject, content, date) values(%s,%s,%s,%s,%s)""",(x ,str(k['From'].split('<')[0][:-1]), str(k['Subject']), str(k['content']), str(dt)))
        print()
        cursor.execute("""INSERT into mails (thread_no, author, subject, content, date) values(%s,%s,%s,%s,%s)""",(x ,str(k['From'].split('<')[0][:-1]), str(k['Subject']), str(k['content']), str(dt)))
        db.commit()
        cnt += 1


db.close()