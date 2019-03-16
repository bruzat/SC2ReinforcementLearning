import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from tensorflow.keras.utils import plot_model

class Logger(object):
    def __init__(self, log_it=20):
        self.it_log = 0
        self.log = ''
        self.log_it = log_it

    def print_train_result(self, epoch, result, score):
        print("__________________________")
        print("| loss 		"+str(result[0]))
        print("| entropy	"+str(result[1]))
        print("| reward	"+str(result[2]))
        print("| score         "+str(score))
        print("| epoch		"+str(epoch))
        print("__________________________")

    def log_train_result(self, path, method, model, epoch, score, result, force = False):
        self.log = self.log +str(epoch)+','+str(result[0])+','+str(result[1])+','+str(result[2])+','+str(score)+'\n'
        self.it_log += 1

        if self.it_log >= self.log_it or force == True:
            writepath = path+'/'+method+'/'+model+'/log.txt'
            os.makedirs(os.path.dirname(writepath), exist_ok=True)
            mode = 'a' if os.path.exists(writepath) else 'w'
            with open(writepath,mode) as file:
                file.write(self.log)
            self.log = ''
            self.it_log = 0

    def drawModel(self,model,path, method_name, model_name):
        writepath = path+'/'+method_name+'/'+model_name+'/model.png'
        plot_model(model, to_file=writepath)
