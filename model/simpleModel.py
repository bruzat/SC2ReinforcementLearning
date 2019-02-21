

import os

from abc import ABCMeta, abstractmethod


class SimpleModel(object):

    def __init__(self):
        pass

    @abstractmethod
    def compile(input_dim, output_dim):
        pass

    def copy(self, model):
        self.model.set_weights(model.get_weights())

    def save(self,writepath):
        os.makedirs(os.path.dirname(writepath), exist_ok=True)
        self.model.save(writepath)

    def load(self,writepath):
        self.model = load_model(writepath)
        self.self_value()

    def predict(self,input):
        return self.model.predict(input)

    def self_value(self):
        self.output = self.model.output
        self.input = self.model.input
        self.trainable_weights = self.model.trainable_weights

    def duplicate_model(self):
        model = self.__class__()
        model.copy(self)
        return model
