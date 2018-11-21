
import json

with open('miniindex.json') as f:
    data = json.load(f)

user_set = set()

for x in data :
    for d in x["messages"] :
    	st=str(d['author'])
    	
    	st=st.split('"')
    	st1=""
    	print("hello")
    	for s in st:
    		print s
    		if not '\"' in s
    			st1=st1+s
    	print(st)
        user_set.add(st1)
        

print(user_set)
for x in data :
	flag=0
    for d in x["messages"] :
    	if flag==0:
