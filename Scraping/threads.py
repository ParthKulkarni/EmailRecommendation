import os, glob, os.path

fpath1 = '/home/vikas/Projects/debian/2019_01/'
fpath2 = '/home/vikas/Projects/debian/2018/'
fpath3 = '/home/vikas/Projects/debian/2017/'
dir_path = '/home/vikas/Projects/debian/debian_dataset/'

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

cnt = 1

def divide_into_thread(path) :
    global cnt
    fpath_ = path + '*/*.txt'
    files = glob.glob(fpath_)
    dict = {}
    for file in files :
        print(file)
        f = open(file, 'r')
        file_msg = f.read()
        msg_lines = file_msg.split('\n')

        for msg_line in msg_lines :
            if msg_line.startswith('Message-id') :
                cur_msg_id = msg_line.split(': ')[1].replace('[🔎]', '').strip()
                par_msg_id = cur_msg_id 
            
            if msg_line.startswith('References :') :
                if len(msg_line.split(':')[1].strip()) != 0 :
                    par_msg_id = msg_line.split('<')[1].split('>')[0].replace('[🔎]', '').strip() 
        
        if dict.get(par_msg_id, '-1') == '-1' :
            dict[par_msg_id] = str(cnt)
            cnt += 1
        temp_dir_path = dir_path + dict.get(par_msg_id)
        # print(temp_dir_path)
        createFolder(temp_dir_path)
        onlyfiles = next(os.walk(temp_dir_path))[2]      
        file_path = temp_dir_path + '/' + str(len(onlyfiles)) + '.txt' 
        f1 = open(file_path, 'w')
        f1.write(file_msg)

# os.makedirs('debian_dataset')
divide_into_thread(fpath3)
divide_into_thread(fpath2)
divide_into_thread(fpath1)
print('done')