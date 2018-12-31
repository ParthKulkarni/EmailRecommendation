import MySQLdb, random

text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."

# Open database connection
db = MySQLdb.connect("localhost","root","1234","flaskapp")

# # checking for 'threads' and 'mails' table in db
# def checkTableExists(dbcon, tablename):
#     dbcur = dbcon.cursor()
#     dbcur.execute("""
#         SELECT COUNT(*)
#         FROM information_schema.tables
#         WHERE table_name = '{0}'
#         """.format(tablename))
#     if dbcur.fetchone()[0] == 1:
#         dbcur.close()
#         return True

#     dbcur.close()
#     return False

# if checkTableExists(db, 'threads') :
#     print("Table - threads exists")
#     dbcur = db.cursor()
#     dbcur.execute("DROP TABLE threads")

# if checkTableExists(db, 'mails') :
#     print("table mails exists")
#     dbcur = db.cursor()
#     dbcur.execute("DROP TABLE mails")

# prepare a cursor object using cursor() method
cursor = db.cursor()

sql = "CREATE TABLE threads(id INT(11) AUTO_INCREMENT PRIMARY KEY, author VARCHAR(25));"
cursor.execute(sql)

# populating 'threads' table
for x in range(15) :
    athr = 'A' + str(x)
    cursor.execute("INSERT INTO threads(author) VALUES(%s);", [athr])

sql = "CREATE TABLE mails(id INT(11) AUTO_INCREMENT PRIMARY KEY, thread_no INT(11), body TEXT);"
cursor.execute(sql)

# populating 'mails' table
for x in range(100) :
    r = int(random.random() * 15)
    cursor.execute("INSERT INTO mails(thread_no, body) VALUES({first}, 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.');".format(first=r))
    db.commit()
    print("done" + str(cursor.rowcount))


db.close()