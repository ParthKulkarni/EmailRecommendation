#Rearranges all the mails inside the project's folders into a single folder and renames it accordingly

import os
import shutil

folder_list={'Debian_apache','Debian_boot','Debian_project'}
years_list={'2017','2018'}
PATH='/home/anuja/Desktop/Arrange/'
#Path to directory containing project folders
cnt=0
for fl in folder_list:
	os.chdir(PATH+fl+'/')
	for yl in years_list:
		os.chdir(PATH+fl+'/'+yl+'/')
		DIR=PATH+fl+'/'+yl+'/'
		#print(os.listdir(DIR))
		for l in os.listdir(DIR):
			os.chdir(PATH+fl+'/'+yl+'/'+l)
			#print(os.listdir(PATH+fl+'/'+yl+'/'+l))
			#print(len([name for name in os.listdir('.') if os.path.isfile(name)]))
			#cnt+=len([name for name in os.listdir('.') if os.path.isfile(name)])
			for files in os.listdir(PATH+fl+'/'+yl+'/'+l):
				print(files)
				cnt=cnt+1
				src=files
				dest=PATH+'/ALL/'+str(cnt)+'.txt'
				shutil.copyfile(src,dest)
#print total count of mails
print(cnt)


