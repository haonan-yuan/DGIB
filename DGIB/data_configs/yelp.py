import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/original/yelp"
processed_datafile_eva = f"{dataroot}/evasive/yelp"
processed_datafile_poi = f"{dataroot}/poisoning/yelp"

dataset = "yelp"
testlength = 8
vallength = 1
length = 24
shift = 3972
num_nodes = 13095
