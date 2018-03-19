import os

class DefaultModelContainer:

    def __init__(self, model=None):
        self.model = model

    def build_model(self):
        return self.model