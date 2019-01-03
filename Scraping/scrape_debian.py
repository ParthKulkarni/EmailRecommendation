from bs4 import BeautifulSoup
import requests
from pprint import pprint
import re

# URL of page to be scraped
page = requests.get('https://lists.debian.org/debian-devel/2018/01/threads.html')
soup = BeautifulSoup(page.text, 'html.parser')

# Get all links on the page and store in a list
links =[]
for ul in soup.find_all("ul"):
    strong = ul.find('strong')
    a      = strong.find('a')
    link = a.attrs['href'] if "href" in a.attrs else ''
    link = 'https://lists.debian.org/debian-devel/2018/01/' + link
    links.append(link)
pprint(len(links))
    

# # Visit each link in the list and scrape data. Data written to file
x = 0
# file = open('debian.txt','w')

for l in links:
    page1 = requests.get(l)
    soup1 = BeautifulSoup(page1.text, 'html.parser')
    fname = str(x) + '.txt'
    file = open(fname,'w')
    x = x+1

    li = soup1.find_all('li')
    dict = {}
    for i in range(len(li)):
        st = li[i].text
        s   = str(st).split(': ')
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
    
    
    
