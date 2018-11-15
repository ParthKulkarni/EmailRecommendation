import sklearn, time, glob, email
import pandas as pd

start_time = time.time()

fpath = "temp_data/All/*.email"
files = glob.glob(fpath)
df = pd.DataFrame(columns=['From', 'To', 'Subject', 'Payload'])
cnt = 1

for file in files :
    f = open(file, 'r')
    mailmsg = f.read()
    msg = email.message_from_string(mailmsg)
    mfrom = msg['From'].split('<')[0]
    mto = msg['To'].split('<')[0]
    msub = msg['Subject']
    mpayload = msg.get_payload()
    if cnt < 2 :
        print (mpayload)
    cnt += 1
    df = df.append({'From': mfrom, 'To': mto, 'Subject': msub, 'Payload': mpayload}, ignore_index = True)

# print(df.head())

print("TIme elapsed : %.7f secs" % (time.time() - start_time))