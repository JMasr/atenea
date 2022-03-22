from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import dlib
import glob
import io

import numpy as np
import scipy.io
import h5py
from sklearn import preprocessing

from pydrobert.kaldi.feat import command_line
from pydrobert.kaldi.io.util import infer_kaldi_data_type
from six.moves import cPickle as pickle
from pydrobert.kaldi.io import open as io_open
from pydrobert.kaldi.io import open as kaldi_open

if len(sys.argv) != 4:
    print(
        "Call this program like this:\n"
        "   ./exec_chinesewhispers_clustering_speechdiar.py arkFiledescriptorsTest thCW output_folder\n"
        "  ")
    exit()

arkFiledescriptorsTest = sys.argv[1]
thCW = float(sys.argv[2])
output_folder_path = sys.argv[3]

descriptors1 = []
keys1 = []
descriptorsArray1 = []

specifier = 'ark' + ':' + arkFiledescriptorsTest
print("File1: {}".format(specifier))
reader1 = kaldi_open(specifier, 'bv')
for idx, tup in enumerate(reader1.items()):
    key, value = tup
    a=np.array(value)
    descriptor = dlib.vector(a.reshape((200,1)))
    descriptors1.append(descriptor)
    keys1.append(key)
    descriptorsArray1.append(a.reshape(200,1))

# exec chinese whispers clustering
labels1 = dlib.chinese_whispers_clustering(descriptors1, thCW)
num_classes1 = len(set(labels1))

# Loop over the classes
file_out = os.path.join(output_folder_path, os.path.basename(arkFiledescriptorsTest)[:-4] + "_clustCW_th" + str(thCW) + ".txt")

print("Th: {}, Number of clusters (Test): {}".format(thCW, num_classes1))
print("File out1: {}".format(file_out))
print("Number of clusters (TH={}): {}".format(thCW,num_classes1))

f = io.open(file_out, 'w', encoding='utf8')
for j, label in enumerate(labels1):
    title = "ID #{} - UTT {}".format(label, keys1[j])
    f.write(title+ '\n')

f.close()
