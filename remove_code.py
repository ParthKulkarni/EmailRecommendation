fpath = "3.email"
f = open(fpath, "r")
msg = f.read()

def remove_code(msg) :
	msg1 = ''
	cnt = 0
	list = ['func', 'import']
	for char in msg :
		if char == '{' :
			cnt += 1
		if char == '}' :
			cnt -= 1
		elif cnt == 0 :
			msg1 += char
		else :
			continue
	msg = msg1
	msg = msg.splitlines()
	msg1 = ''
	for line in msg :
		if line.strip() == '' :
			print("*******")
			continue
		flag = True
		for x in list :
			if x in line :
				flag = False
		if flag :
			msg1 += line + '\n'
	print (msg1)

remove_code(msg)