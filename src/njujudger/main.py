import json
import os
import multiprocessing

# from predictor import Predictor
from judger import Judger
import sys

data_path = "input_path"  # The directory of the input data
output_path = "output_path"  # The directory of the output data


def format_result(result):
    rex = {"accusation": [], "articles": [], "imprisonment": -3}

    res_acc = []
    for x in result["accusation"]:
        if not (x is None):
            res_acc.append(int(x))
    rex["accusation"] = res_acc

    if not (result["imprisonment"] is None):
        rex["imprisonment"] = int(result["imprisonment"])
    else:
        rex["imprisonment"] = -3

    res_art = []
    for x in result["articles"]:
        if not (x is None):
            res_art.append(int(x))
    rex["articles"] = res_art

    return rex


if __name__ == "__main__":
#     user = Predictor()
    accusation_path = './predictor/constant/accu.txt'
    law_path = './predictor/constant/law.txt'
    truth_path,output_path = './predictor/input/','./predictor/output/'
    judger = Judger(accusation_path,law_path)
    result = judger.test(truth_path,output_path)
    score = judger.get_score(result)
    print(result)
    print(score)

