import os

class Logger(object):
    def __init__(self, log_it=20):
        self.it_log = 0
        self.log = ''
        self.log_it = log_it

    def print_train_result(self, epoch, result):
        print("__________________________")
        print("| loss 		"+str(result[0]))
        print("| entropy	"+str(result[1]))
        print("| reward	"+str(result[2]))
        print("| epoch		"+str(epoch))
        print("__________________________")

    def log_train_result(self, path, model, epoch, result, force = False):
        self.log = self.log + str(epoch)+','+str(result[0])+','+str(result[1])+','+str(result[2])+'\n'
        self.it_log += 1

        if self.it_log >= self.log_it or force == True:
            writepath = './'+path+'/'+model+'/log.txt'
            os.makedirs(os.path.dirname(writepath), exist_ok=True)
            mode = 'a' if os.path.exists(writepath) else 'w'
            with open(writepath,mode) as file:
                file.write(self.log)
            self.log = ''
            self.it_log = 0
