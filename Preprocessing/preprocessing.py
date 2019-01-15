import glob
# from flanker import mime
import re
import spacy
from langdetect import detect
import nltk
from nltk.tag.stanford import StanfordNERTagger


class preprocess:

	def remove_content_in_braces(self, msg):
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

	def remove_func_and_struct(self,msg):
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
				if len(words)  >= 3  and words[2] == "struct" :
					take_line = False
			if take_line :
				msg1 += (line + '\n')
		return msg1


	def remove_other_code_lines(self,msg) :
		msg1 = ''
		take_line = True
		msg = msg.splitlines()
		i = 0	
		while i < len(msg) :
			if detect(str(msg[i])) != 'en':
				continue  
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


	def remove_code(self, msg) :
		msg = self.remove_content_in_braces(msg)
		msg = self.remove_func_and_struct(msg)
		msg = self.remove_other_code_lines(msg)
		return msg
		                   
	def replace_tokens(self, message):
		message = re.sub(r"\w*.doc$|\w*.pdf$|\w*.txt$|\w*.xls$|\w*.ppt$", "", message) 
		message = re.sub(r"\swhy$|\swhere$|\swho$|\swhat$|\swhen$","", message)  
		message = re.sub(r"(http|https|ftp)://.*|www.*", "", message)
		message = re.sub(r"\S+@\S+", "", message) 
		message = re.sub(r"\smonday|\smon|\stuesday|\stue|\swednesday|\swed|\sthursday|\sthu|\sfriday|\sfri|\ssaturday|\ssat|\ssunday|\ssun", "", message)
		message = re.sub(r"\sme$|\sher$|\shim$|\sus$|\sthem$", "", message)
		message = re.sub(r"\sI$|\swe$|\syou$|\she$|\sshe$|\sthey$", "", message)
		message = re.sub(r'\d+', "" ,message)
		message = re.sub(' +', ' ', message)
		message = re.sub("(,|;|\+|\-|\$|=|<|>|[|]|\*|`|\"|:|/)+", "", message)

		return message

	def lemmatize(self, doc):
		nlp = spacy.load("en")
		my_stop_words = ['gmail','google','github','yahoo','com','org','http','https','golang','www','regards','thanks','html5','tidyhtml5']
		for stopword in my_stop_words:
			lexeme = nlp.vocab[stopword]
			lexeme.is_stop = True
		doc = nlp(doc)
        # we add some words to the stop word list
		texts, article, skl_texts = '','',''
		for w in doc:
            # if it's not a stop word or punctuation mark, add it to our article
			if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and w.text!='I':
#               # we add the lemmatized version of the word
				if w.lemma_!='-PRON-' and w.lemma_!=' \n'and w.lemma_!='\n \n \n':
					article += " " + w.lemma_
            # if it's a new line, it means we're onto our next document
			if w.text == '\n':
				skl_texts += " " + article
				texts += " " + article
				article = ''
		return skl_texts

	# def extract_mail_body(self, msg):
	# 	rt = ''
	# 	msg = mime.from_string(msg)
	# 	if msg.content_type.is_singlepart():
	# 		temp = str(msg.body)
	# 		temp = temp.splitlines()
	# 		for _ in temp:
	# 			if _.startswith('>'):
	# 				continue
	# 			elif _.startswith('On'):
	# 				continue
	# 			else:
	# 				rt+=_+"\n"
	# 	else :
	# 		for part in msg.parts :
	# 			if "(text/plain)" in str(part) :
	# 				temp = str(part.body)
	# 				temp = temp.splitlines()
	# 				for _ in temp :
	# 					if _.startswith('>') :
	# 					  continue
	# 					if _.startswith('On'):
	# 					  continue
	# 					else :
	# 					  rt+=_+"\n"
	# 	return rt


if __name__ == '__main__':
    
    obj = preprocess()

    files = glob.glob("000_4Ce8YN2m-AwJ.email")

    bodies =[]
    for file in files:
        f = open(file, "r")
        msg = f.read()
        body = obj.extract_mail_body(msg)
        bodies.append(body)
    #print(bodies)

    preprocessed = []
    for body in bodies:
        nocode = obj.remove_code(body)
        #print(nocode)
        lemm = obj.lemmatize(str(nocode))
        #print(lemm)
        replaced = obj.replace_tokens(str(lemm))
        # print(replaced)
        preprocessed.append(replaced)
        # print(replaced)
        # print('************')            
		# print(replaced)
    print(preprocessed)

