import model

md = model.model()
li = md.generate_random_list()

print(li)

def list_to_string(l) :
    s = ''
    for item in l :
        s = s + "'" + str(item) + "',"
    s = s[:-1]
    return s

str_li = str(list_to_string(li))
print(str_li)

print("SELECT * FROM threads WHERE id in ({first})".format(first=str_li))