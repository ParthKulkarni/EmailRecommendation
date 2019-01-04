from bs4 import BeautifulSoup
import requests
from pprint import pprint
import re

import requests
from bs4 import BeautifulSoup
import urllib
 
url = 'https://lists.debian.org/debian-devel/'
year = 2019
month = 0
url1 = 'threads.html'

links = []
def getLinks(url, url1):
	try :
		html_page = urllib.request.urlopen(url + url1)
		soup = BeautifulSoup(html_page, "lxml")
		links1 = [a.get('href') for a in soup.find_all('a', href=True)]
		# links = []
		for link in links1 :
			if link.startswith('msg') :
				links.append(url + link)
		print(len(links)
	except urllib.error.HTTPError as err:
		if err.code == 404 :
			print()
		else :
			raise


for x in range(1) :
	month += 1
	base_url = ''
	if month < 10 :
		base_url = url + str(year) + '/0' + str(month) + '/'
	else :
		base_url = url + str(year) + '/' + str(month) + '/'
	# print(base_url + url1)
	getLinks(base_url, url1)
	for y in range (2, 5) :
		extra_url = 'thrd' + str(y) + '.html'
		# print(base_url + extra_url)
		getLinks(base_url, extra_url)
	
    

# # Visit each link in the list and scrape data. Data written to file
x = 0
# file = open('debian.txt','w')

for l in links:
    page1 = requests.get(l)
    soup1 = BeautifulSoup(page1.text, 'html.parser')
    fname = str(x) + '.txt'
    file = open(fname,'w')
    x += 1

    li = soup1.find_all('li')
    dict = {}
    for i in range(len(li)):
        st = li[i].text
        s  = str(st).split(': ', 2)
        if s[0] == 'Cc':
            continue
        dict[s[0]] = s[1]
        if s[0] == 'Message-id':
            break
    dict['Message-id']=dict['Message-id'][5:-1]
    for n,m in dict.items():
        file.write(n + ' : '+m+'\n')

    pre = soup1.find_all('pre')
    tt  = soup1.find_all('tt')
    if len(pre)>0:
        print(len(pre))
        body = pre[0].text
    for t in tt:
        body += t.text
    if len(pre) > 1:
        body += pre[-1].text
    body = re.sub('\n+', '\n',body)
    body = body.lstrip()
    body = body.rstrip()
    # print(body)
    file.write(body)
    file.close()
    
    
    
