class file_content :
    
    def __init__(self, path) :
        self.f_pointer = open(path, 'r')
        self.file_msg = self.f_pointer.read()
        # self.To = ''
        # self.From = ''
        # self.Subject = ''
        # self.Date = ''
        # self.Message-id = ''
        # self.In-reply-to = ''
        # self.References = ''
        # self.root_mail = ''
        # self.content = ''
        self.mail = {}

    def read_file_content(self) :
        content = self.file_msg.split('\n\n\n')
        header = content[0]
        header_lines = header.splitlines()
        for line in header_lines :
            line = line.split(':', 1)
            key = line[0].strip()
            value = line[1].replace('[🔎]', '').strip()
            self.mail[key] = value
        self.mail['content'] = content[1].replace('[🔎]', '').strip()

        # for key, value in self.mail.items() :
        #     print(key + ' :-> ' + value)

# fpath = '/home/vikas/Projects/debian/2019_01/1/0.txt'

# ob = file_content(fpath)
# ob.read_file()
