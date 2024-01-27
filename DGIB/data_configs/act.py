import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/original/act"
processed_datafile_eva = f"{dataroot}/evasive/act"
processed_datafile_poi = f"{dataroot}/poisoning/act"

dataset = "act"
testlength = 8
vallength = 2
length = 30
