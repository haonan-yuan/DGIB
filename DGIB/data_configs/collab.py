import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
dataroot = os.path.join(CUR_DIR, "../../data")
processed_datafile_ori = f"{dataroot}/original/collab"
processed_datafile_eva = f"{dataroot}/evasive/collab"
processed_datafile_poi = f"{dataroot}/poisoning/collab"

dataset = "collab"
testlength = 5
vallength = 1
length = 16
