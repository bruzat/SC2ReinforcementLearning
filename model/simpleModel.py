from tensorflow.keras import Model

import os

class SimpleModel(object):

    def __init__(self):
        super().__init__()
        self.model = None
        self.input_dim = None
        self.output_dim = None

    def compile(self,input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None

    def save(self,writepath):
        os.makedirs(os.path.dirname(writepath), exist_ok=True)
        self.model.save(writepath)

    def load(self,writepath):
        self.model = Model.load_model(writepath)
        self.self_value()

    def predict(self,input):
        return self.model.predict(input)

    def self_value(self):
        self.output = self.model.output
        self.input = self.model.input
        self.trainable_weights = self.model.trainable_weights

    def copy(self, model):
        self.model.set_weights(model.model.get_weights())

    def duplicate_model(self):
        model = self.__class__()
        model.compile(self.input_dim,self.output_dim)
        model.copy(self)
        return model
