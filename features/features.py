from pprint import pprint
from flanker import mime
import json
import numpy
import pickle
import re
import csv
from email.utils import parsedate_tz, mktime_tz, formatdate
import time


with open('miniindex.json') as f:
    data = json.load(f)
cnt=0
user_set = set()
us=0

#Assign a userid to each user and store it in user_set from index.json

for x in data :
	for d in x["messages"] :
		st=str(d['author'])
		st=st.split('"')
		st1=""
		for s in st:
			st1=st1+s
		user_set.add(str(st1))
		


#userlist (username==>userid) gives userid
userlist={} 
#usrmail (messageid==>userid) gives author
usr_mail={}
# u=number of users
u=len(user_set)
mat = [[0 for x in range(u)] for y in range(u)] 

i=0
with open('/home/anuja/Desktop/BE project/features/CSV/userinfo.csv', 'w', newline='') as csvfile:
	fieldnames = ['userid', 'uname']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()  
	for u in user_set:
		userlist.update({str(u) : i})
		writer.writerow({'userid': str(i), 'uname': str(u)})
		i=i+1
		
print("\n\n\nTotal number of users = "+str(len(user_set))+" \nuserlist:==> ")
print(userlist)

#mails= total mails read
mails=0

#mails.csv ==> stores info of each mail
with open('/home/anuja/Desktop/BE project/features/CSV/mails.csv', 'w', newline='') as csvfile:
	fieldnamess = ['mailid','authorid','timestamp', 'main/reply','msglen','qcnt','whcnt','file','text']
	mailwriter = csv.DictWriter(csvfile, fieldnames=fieldnamess)
	mailwriter.writeheader()
	#thread.csv ==> stores reply msgid and which msgid it replies to
	with open('/home/anuja/Desktop/BE project/features/CSV/thread.csv', 'w', newline='') as file1:
		fieldnames1 = ['reply_mailid','inreplyto_mailid']
		threadwriter = csv.DictWriter(file1, fieldnames=fieldnames1)
		threadwriter.writeheader()
		#for each thread
		for x1 in data:
			#for each mail in thread
			for d1 in x1["messages"] :
				head=dict()	
				filename=str('/home/anuja/Desktop/BE project/features/Dataset/'+d1['file'])
				file = open(filename, "r")
				cnt=cnt+1
				mailmsg = file.read()
				msg = mime.from_string(mailmsg)
				text =' '
				qcnt=0
				wh_cnt=0
				wh_quest={'what','who','where','why','when','how','which','whose','whom'}

				if msg.content_type.is_singlepart():
					temp = str(msg.body)
					temp = temp.splitlines()
					for _ in temp:
						if _.startswith('>'):
							continue
						elif _.startswith('On'):
							continue
						else:
							text+=_
							
				else :
					for part in msg.parts :
						if "(text/plain)" in str(part) :
							temp = str(part.body)
							temp = temp.splitlines()
							for _ in temp :
							    if _.startswith('>') :
							        continue
							    if _.startswith('On'):
							        continue
							    else :
							        text+=_
							        
							        
				
				qcnt=text.count('?')
				for wh in wh_quest:
					wh_cnt=wh_cnt + text.count(wh)
	        	
				#print("Wh_questions: "+str(wh_cnt))
				#print("Qcnt: "+str(qcnt))
				#print("Text Body \n"+text)
				head=dict(msg.headers.items())
				date=str(head["Date"]).split('(')[0]
				#date = parsedate_tz(date)
				#timestamp = mktime_tz(date)
				#print(formatdate(timestamp))
				
				#mailid ==> message id of mail
				mailid=str(head["Message-Id"].replace('<',''))
				mailid=str(mailid.replace('>',''))
				

				if 'In-Reply-To' in head:
					#print("Reply message")
					#replymailid ==> message id which themail replies to
					replymailid=str(head["In-Reply-To"].replace('<',''))
					replymailid=str(replymailid.replace('>',''))
					#print("ReplyMailid "+replymailid)
					
					#repl_id==> usrid who is replying
					rep=head["From"]
					rep=str(re.sub('<.*?>', '', str(rep)))
					rep=rep.split('"')
					repl="" 
					for r in rep:
						repl=repl+r
					repl=repl.split('\\')
					rep="" 
					for r in repl:
						rep=rep+r
					repl_id=0
					for k,v in userlist.items():
						k=k.strip()
						rep=rep.strip()
						if str(k) == str(rep):
							repl_id=v
					
			
					usr_mail.update({str(mailid) : str(repl_id)})
					
					mailwriter.writerow({'mailid':mailid,'authorid':repl_id,'file':filename ,'text': str(text),'main/reply':0,'qcnt':qcnt,'whcnt':wh_cnt,'msglen':str(len(text)),'timestamp':date})
					threadwriter.writerow({'reply_mailid':str(mailid),'inreplyto_mailid':str(replymailid)})
					for k,v in usr_mail.items():
						if str(k) == str(replymailid):
							replyto_id=int(v)
					#print("ReplyId "+str(repl_id)+" in reply to "+str(replyto_id))	
					mat[repl_id][replyto_id]=mat[repl_id][replyto_id]+1
					mails=mails+1
				else:
					#print("Main message")
					ini=head["From"]
					ini=ini.split('<')[0]
					ini=ini.split('"')
					init="" 
					for r in ini:
						init=init+r
					init=init.split('\\')
					ini="" 
					for r in init:
						ini=ini+r
					owner=ini
					#print("Initiated by "+owner)
					for k,v in userlist.items():
						k=k.strip()
						owner=owner.strip()
						if str(k) == str(owner):
							own_id=v
					mailwriter.writerow({'mailid': str(mailid),'authorid':own_id,'file':filename , 'text': str(text),'qcnt':qcnt,'main/reply':1,'whcnt':wh_cnt,'msglen':str(len(text)),'timestamp':date})
					usr_mail.update({str(mailid) : str(own_id)})
					mails=mails+1

#matrix row=>userid col=>usr who replied
#for mat[i][j]=k
# jth usr replied to ith usr k times				
#print(mat)			
num_mat=numpy.matrix(mat)
num_mat.dump("mymatrix.dat")

new_mat=numpy.load("mymatrix.dat")
print(new_mat)
print("Total mails analyzed:"+str(cnt))
