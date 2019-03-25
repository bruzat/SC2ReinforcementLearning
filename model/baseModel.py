from tensorflow.keras import models
from tensorflow.keras.models import model_from_json

import os

class BaseModel(object):

    def __init__(self):
        super().__init__()
        self.model = None
        self.input_dim = None
        self.output_dim = None
        self.activation = None

    def make(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.activation = activation

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_json = self.model.to_json()
        with open(path+".json", "w+") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(path+".h5")

    def load_model(self, path):
        json_file = open(path+".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(path+".h5")

    def predict(self,env):
        return self.model.predict(env)

    def self_value(self):
        self.output = self.model.output
        self.input = self.model.input
        self.trainable_weights = self.model.trainable_weights

    def copy(self, model):
        self.model.set_weights(model.model.get_weights())

    def duplicate_model(self):
        model = self.__class__()
        model.make(self.input_dim,self.output_dim)
        model.copy(self)
        return model

    def compile(self,optimizer,loss):
        self.model.compile(optimizer=optimizer, loss=loss)
