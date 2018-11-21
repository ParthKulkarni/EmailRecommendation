from pprint import pprint
from flanker import mime
import json
import numpy
import pickle
import re

with open('mini_index.json') as f:
    data = json.load(f)
cnt=0
user_set = set()

for x in data :
    for d in x["messages"] :
    	#st=str(d['author'])
    	st=d['author'].encode('utf-8').strip()
    	st=st.split('"')
    	st1=""
    	for s in st:
    		st1=st1+s
    	#print(st1)
        user_set.add(str(st1))
#print(user_set)

userlist={}
invlist={}
i=0



w, h = 40, 40;
mat = [[0 for x in range(w)] for y in range(h)] 

i=0  
for u in user_set:
	userlist.update({i : str(u)})  #index gives uname
  	invlist.update({str(u) : i}) #uname gives index
  	i=i+1
print("userlist==>")
print(userlist)
print("invlist:==> ")
print(invlist)

for x1 in data:
	flag=0
	owner=""
	for d1 in x1["messages"] :
		head=dict()	
		file = open(str('/home/anuja/Desktop/BE project/features/'+d1['file']), "r")
		cnt=cnt+1
		mailmsg = file.read()
		msg = mime.from_string(mailmsg)
		head=dict(msg.headers.items())
		#print(head)
		if 'In-Reply-To' in head or 'Reply-To' in head:
			print("Reply message")
			if 'To' in head:
				send=head["To"]
			elif 'In-Reply-To' in head:
				send=head["In-Reply-To"]
			elif 'Reply-To' in head:
				send=head["Reply-To"]
			send=str(re.sub('<.*?>', '', str(send)))
			send=send.split('"')
			sendr="" 
			for ss in send:
				sendr=sendr+ss
			sendr=sendr.split('\\')
			send="" 
			for ss in sendr:
				send=send+ss
			#print(send)
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
			send_id=0
			for k,v in invlist.iteritems():
				k=k.strip()
				rep=rep.strip()
				if str(k) == str(rep):
					repl_id=v
			
			f=0
			for k,v in invlist.iteritems():
				k=k.strip()
				send=send.strip()
				if str(k) == str(send):
					send_id=v
					f=1
					print("*Replied to=> "+send+" id "+str(send_id))
					
			if f==0:
			 for k,v in invlist.iteritems():
				k=k.strip()
				send1=owner
				send1=send1.strip()
				if k == send1:
					send_id=v
					print("Replied to=> "+send+" id "+str(send_id))
			
				
			print("Replier=> "+rep+" id "+str(repl_id))
			mat[send_id][repl_id]=mat[send_id][repl_id]+1
		else:
			print("Main message")
			ini=head["From"]
			#ini=str(re.sub('<.*?>', '', str(ini)))
			#ini=re.sub('<.*?>', '', str(ini)).encode('utf-8').strip()
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
			print("Initiated by "+owner)
print(mat)			
num_mat=numpy.matrix(mat)
num_mat.dump("mymatrix.dat")

new_mat=numpy.load("mymatrix.dat")
print(new_mat)
print("Total mails analyzed:"+str(cnt))