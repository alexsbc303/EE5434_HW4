# colors = ['tab:purple', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:blue', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
COLORS = ['purple', 'orange', 'green', 'red', 'brown', 'blue', 'pink', 'gray', 'olive', 'cyan']
GROUPS = range(0, 10)
groups = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
MARKERS = [".", "v", "s", "p", "*", "+", "d", "|", 0, 9]

import numpy
import os
import wget

class Digit:
    def __init__(self, name, decimal, symm):
        self.name = name
        self.decimal = decimal
        self.symm = symm
        self.color = COLORS[name]
        self.group = name

    def __str__(self):
        return '[%s: decimal(%4f), symm(%4f), color(%s), group(%s)]' % (self.name, self.decimal, self.symm, self.color, self.group)


class Dataset:
    def __init__(self):
        self.data = [None] * 10
        self.d = []

    def read_data(self, path):
        file = open(path, "r")
        lines = file.read().splitlines()
        for line in lines:
            (_, _, _, name, _, _, decimal, _, symm) = line.split(' ')
            name = int(float(name))
            if self.data[name] is None:
                self.data[name] = []
            self.data[name].append({'name': name, 'decimal':float(decimal), 'symm':float(symm)})
        return self.data

    def read_raw_1and5(self, path):
        file = open(path, "r")
        lines = file.read().splitlines()
        for line in lines:
            tokens = line.split(' ')[:-1]
            if int(float(tokens[0])) == 1 or int(float(tokens[0])) == 5:
                l = []
                for token in tokens:
                    l.append(token)
                self.d.append(l)
        raw = numpy.asarray(self.d)
        raw_yaxis = raw[:,0].astype('float32').astype('int')
        raw_xaxis = raw[:,1:].astype('float32')
        return raw_xaxis, raw_yaxis

    def __str__(self):
        for each in self.data:
            print(each)

def download_data():
    # Download data
    if not os.path.isfile('zip.train'):
        print('Downloading Data...')
        urls = ['http://amlbook.com/data/zip/zip.train',
                'http://amlbook.com/data/zip/zip.test',
                'http://amlbook.com/data/zip/features.train',
                'http://amlbook.com/data/zip/features.test']
        for url in urls:
            wget.download(url)
    else:
        print('Data already exists.')