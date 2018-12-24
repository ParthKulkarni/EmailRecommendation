from bs4 import BeautifulSoup
import requests

# URL of page to be scraped
page = requests.get('https://lkml.org/lkml/2018/1/1')
soup = BeautifulSoup(page.text, 'html.parser')

# Get all links on the page and store in a list
links =[]
for anchor in soup.find_all("a"):
    link = anchor.attrs["href"] if "href" in anchor.attrs else ''
    link = 'https://lkml.org' + link
    links.append(link)
links.pop()
links = links[7:]

# Visit each link in the list and scrape data. Data written to file
for l in links:
    page1 = requests.get(l)
    soup1 = BeautifulSoup(page1.text, 'html.parser')

    last_links = soup1.find(class_='c')
    last_links.decompose()
    name_list = soup1.findAll(class_='lp')
    name_list_items = soup1.findAll(class_= 'rp')
    body = soup1.findAll(itemprop='articleBody')

    body = str(body)
    fname = l[31:]
    fname = fname +'.email'
    file = open(fname,'w')

    for i in range(len(name_list)):
        file.write(name_list[i].contents[0] + ' : ' + name_list_items[i].contents[0] + '\n')
    
    body = body.replace('<br/>','\n')
    file.write(body)
    file.close() 
    
    
