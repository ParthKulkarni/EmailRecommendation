import random


class model :
    def __init__(self) :
        self.list_to_predict = []


    def generate_random_list(self) :
        li = []
        li_name = []
        for x in range (10) :
            li.append((random.randint(0, 100)) % 2)

        for x in range (10) :
            if li[x] == 1 :
                li_name.append(x)
        return li_name


# print(model().generate_random_list())