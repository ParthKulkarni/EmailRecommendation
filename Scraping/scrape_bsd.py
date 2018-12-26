from bs4 import BeautifulSoup
import requests
from pprint import pprint
import re

page = requests.get('https://marc.info/?l=aic7xxx&m=132745954721164&w=2')
soup = BeautifulSoup(page.text, 'html.parser')

data = soup.findAll(text=True)

r = ''
for d in data:
    r += str(d)
r = r.replace('[Download RAW message or body]','')
r = re.sub('\n+', '\n',r)
r = r.lstrip()
r = r.rstrip()
r = r.split('\n',3)[3]
r = r.rsplit('\n',7)[0]

file = open('bsd.txt','w')
r = r.split('\n',4)
for i in range(4):
    r[i] = re.sub(' +', ' ',r[i])
    s = r[i].split(': ')
    file.write(str(s))
file.write('\n')
file.write(r[4])
file.close()