from flanker import mime

fpath = "3.email"

f = open(fpath, 'r')
mailmsg = f.read()
msg = mime.from_string(mailmsg)

# print (msg.headers.items())

print("printing email message!!")
if msg.content_type.is_singlepart():
    temp = str(msg.body)
    temp = temp.splitlines()
    for _ in temp:
        if _.startswith('>'):
            continue
        else:
            print("*** " + _)
    print("********************************")
elif  msg.content_type.is_multipart() :
    for part in msg.parts :
        if "(text/plain)" in str(part) :
            temp = str(part.body)
            temp = temp.splitlines()
            for _ in temp :
                if _.startswith('>') :
                    continue
                else :
                    print("*** " + _)
            print("********************************")
