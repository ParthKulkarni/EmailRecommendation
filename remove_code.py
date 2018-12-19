import re

fpath = "/home/niki/Documents/BE_Project/EmailRecommmendation/temp_data/Code_Snippets/000_OJB7kVtYOzEJ.email"
f = open(fpath, "r")
msg = f.read()

def remove_content_in_braces(msg) :
	msg1 = ''
	cnt = 0
	for char in msg :
		if char == '{' :
			cnt += 1
		elif char == '}' :
			cnt -= 1
		elif cnt == 0 :
			msg1 += char
		else :
			continue
	return msg1

def remove_func_and_struct(msg) :
	msg1 = ''
	take_line = True
	msg = msg.splitlines()
	for line in msg :
		take_line = True
		if line == '' :
			continue
		words = line.split(' ')
		if words[0] == "func" :
			take_line = False
		elif words[0] == "type" :
			if words[2] == "struct" :
				take_line = False
		if take_line :
			msg1 += (line + '\n')
	return msg1

def remove_other_code_lines(msg) :
	msg1 = ''
	take_line = True
	msg = msg.splitlines()
	i = 0	
	while i < len(msg) :
		if (msg[i] == '') or ("//" in msg[i]) :
			i += 1
			continue
		take_line = True
		line = msg[i]
		if "package" in line :
			words = line.split(' ')
			if len(words) < 4 :
				take_line = False
		elif "import" in line :
			words = line.split(' ')
			if len(words) < 4 :
				take_line = False
				if "(" in line :
					while ')' not in msg[i] :
						i += 1
		elif "const" in line :
			words = line.split(' ')
			if len(words) < 4 :
				take_line = False
				if "(" in line :
					while ')' not in msg[i] :
						i += 1
		if take_line :
			msg1 += line + '\n'
		i += 1
	return msg1


def remove_code(msg) :
	msg = remove_content_in_braces(msg)
	msg = remove_func_and_struct(msg)
	msg = remove_other_code_lines(msg)
	print(msg)
	

remove_code(msg)