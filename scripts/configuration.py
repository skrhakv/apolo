import json
import os

class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Configuration(object):
    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return Configuration.load_dict(data)
        elif type(data) is list:
            return Configuration.load_list(data)
        else:
            return data

    @staticmethod
    def load_dict(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = Configuration.__load__(value)
        return result

    @staticmethod
    def load_list(data: list):
        result = [Configuration.__load__(item) for item in data]
        return result

    @staticmethod
    def load_json(filename=None):
        if filename == None:
            filename = 'configurations/configuration.json'
        path_dir = os.path.realpath(os.path.dirname(__file__))
        path = f'{path_dir}/../{filename}'
        
        with open(path, "r") as f:
            result = Configuration.__load__(json.loads(f.read()))
        return result
    
