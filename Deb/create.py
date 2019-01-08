#Reads mails from /1 Folder and groups the mails as per their threads in separate folders in Threads/ directory
#Mails are ordered as per their timestamp


import os
import shutil
import csv
import datetime
from flanker import mime

PATH='/home/anuja/Desktop/Deb/'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

main_set=set()
thread_dict={}
list_dict={}
i=0


TOTAL=801 #Total number of mails to be grouped


with open(PATH+'thread.csv', 'w', newline='') as file1:
		fieldnames1 = ['threadid','Subject']
		threadwriter = csv.DictWriter(file1, fieldnames=fieldnames1)
		threadwriter.writeheader()
		#PassOne Group main messages to create folder for each new thread
		for x in range(1,TOTAL):
			print("======================================================================="+str(x))
			file= open(PATH+'1/'+str(x)+'.txt', 'r')
			mailmsg=file.read()
			msg = mime.from_string(mailmsg)
			temp = str(msg.body)
			text=""
			temp = temp.splitlines()
			#print("-----------------------------------------------------------------------")
			for t in temp:
				if t.startswith('>'):
					continue
				elif t.startswith('On'):
					continue
				elif t.startswith('&gt;'):
					continue
				elif t.startswith('+'):
					continue
				elif t.startswith('To'):
					#print(t)
					continue
				elif t.startswith('From'):
					#print(t)
					continue
				elif t.startswith('Subject'):
					t=t.replace("Subject : ","")	
					t=t.replace("Subject: ","")	
					#print(t)
					if t.startswith("Re:"):
						#print("---REPLY")
						continue
					elif t.startswith("RE:"):
						#print("---REPLY")
						continue
					else:
						#print("---Main")
						main_set.add(t)
						t=t.strip()
						#print("--"+t)
						t1=list()
						t1={}
						thread_dict.update({str(t) : str(i)})
						list_dict.update({str(i) :t1 })
						createFolder('./Thread/'+str(i)+'/')
						dest=PATH+'Thread/'+str(i)+'/'+str(1)+'.txt'
						#print("Dest: "+dest)
						src=PATH+'1/'+str(x)+'.txt'
						#print("src "+src+" dest "+dest)
						shutil.copyfile(src,dest)
						threadwriter.writerow({'threadid':str(i),'Subject':str(t)})
						i=i+1
				elif t.startswith('Date'):
					#print(t)
					continue
					date=str(t).split('(')[0]
				elif t.startswith('Message-id'):
					continue
				else:
					text=text+t
			
		#PassTwo Group reply messages 'RE:' in respective folders acc to 'Subject:'
		#Rename the thread messages as per their timestamp
		for x in range(1,TOTAL):
			print("======================================================================="+str(x))
			file= open(PATH+'1/'+str(x)+'.txt', 'r')
			mailmsg=file.read()
			#print(mailmsg)
			msg = mime.from_string(mailmsg)
			temp = str(msg.body)
			print(str(x)+'.txt')
			text=""
			temp = temp.splitlines()
			date=""
			for t in temp:
				if t.startswith('Date :'):
					#print(t)
					date=str(t).split('Date :')[1]
				elif t.startswith('Date:'):
					#print(t)
					date=str(t).split('Date:')[1]
					
					
			for t in temp:
				if t.startswith('>'):
					continue
				elif t.startswith('On'):
					continue
				elif t.startswith('&gt;'):
					continue
				elif t.startswith('+'):
					continue
				elif t.startswith('To'):
					#print(t)
					continue
				elif t.startswith('From'):
					#print(t)
					continue
				if t.startswith('Subject'):
					t=t.replace("Subject : ","")	
					
					
					if t.startswith("Re:"):
						print(date.split('(')[0])
						date=date.split('(')[0]
						date=date.rstrip()
						
						print(date)
						tt=t.split("Re:")[1]
						dest=PATH+'Thread/0/'+str(x)+'.txt'
						tt=tt.strip()
						#print("++"+tt)
						main=0
						for k,v in thread_dict.items():
								#print("key "+k+" val "+v+" "+tt)
								if str(k)==str(tt):
									dest=PATH+'Thread/'+v+'/'+str(date)+'.txt'
									main=v
						for k,v in list_dict.items():
								if str(k)==str(main):
									temp=list()
									#temp={x}
									temp={date}
									v=list(v)
									v.extend(temp)
									list_dict.pop(k)
									list_dict.update({str(k) :v })
									break
						src=PATH+'1/'+str(x)+'.txt'
						
						shutil.copyfile(src,dest)
					elif t.startswith("RE:"):
						#print(date)
						#print("---REPLY "+t)
						tt=t.split("RE:")[1]
						dest=PATH+'0/'+str(x)+'.txt'
						tt=tt.strip()
						#print("++"+tt)
						
						for k,v in thread_dict.items():
								#print("key "+k+" val "+v+" "+tt)
								if str(k)==str(tt):
									#dest='/home/anuja/Desktop/BE project/Data3/Thread/'+v+'/'+str(x)+'.txt'
									dest=PATH+'Thread/'+v+'/'+str(date)+'.txt'
									main=v
									#print("Dest: "+dest)
						for k,v in list_dict.items():
								#print("key "+k+" val "+v+" "+tt)
								if str(k)==str(main):
									temp=list()
									#temp={x}
									temp={date}
									v=list(v)
									v.extend(temp)
									list_dict.pop(k)
									list_dict.update({str(k) :v })
									break
						src=PATH+'1/'+str(x)+'.txt'
						shutil.copyfile(src,dest)
				elif t.startswith('Message-id'):
					continue
				else:
					text=text+t
#Sort files in each thread folder acc to timestamp and order the mail files 
with open('thread.csv') as csv_file :
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		threadid=row[0]
		print("---THREAD ID "+str(threadid))	
		
		list1=list_dict.get(str(threadid))
		
		#print(list1)
		if bool(list1)==False:
			 no_of_msg=0
		else:
			list1=list(list1)
			no_of_msg=len(list1)
			print(list1)
			print("After sorting")
			list1.sort(key=lambda date: datetime.datetime.strptime(date,' %a, %d %b %Y %H:%M:%S %z'))
			print(list1)
			cnt=2
			os.chdir(PATH+'Thread/')
			for x in list1:
				if str(threadid)!='0':
					os.chdir(PATH+'Thread/'+str(threadid))
					print(str(threadid)+"/"+x)
					#print(f"We have: {os.listdir()}")
					os.rename(str(x)+'.txt',str(cnt)+'.txt')
					cnt=cnt+1	
			

#Check if sorting is done right
d1=' Tue, 31 Jan 2017 12:38:18 +0100'
d1=datetime.datetime.strptime(d1,' %a, %d %b %Y %H:%M:%S %z')
d2=' Wed, 01 Feb 2017 00:19:33 +1100'
d2=datetime.datetime.strptime(d2,' %a, %d %b %Y %H:%M:%S %z')
d3=' Tue, 31 Jan 2017 12:17:29 -0600'
d3=datetime.datetime.strptime(d3,' %a, %d %b %Y %H:%M:%S %z')
print("Difference is")
print(d2-d1)
print(d3-d2)
