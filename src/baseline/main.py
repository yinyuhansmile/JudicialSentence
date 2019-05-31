import json
import os
import multiprocessing
import time 

from predictor import Predictor

data_path = "./predictor/resource/data_test.json"  # The directory of the input data
output_path = "./predictor/output"  # The directory of the output data


def format_result(result):
    rex = {"accusation": [], "articles": [], "imprisonment": -3}
#     print("result hah",result)
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
    user = Predictor()
    cnt = 0


    def get_batch():
        v = user.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v


    def solve(fact):
        result = user.predict(fact)
        print ("slove!!!")

        for a in range(0, len(result)):
            result[a] = format_result(result[a])

        return result

    start_time = time.time()
    print("starttime:"+str(start_time))
    inf = open(data_path, 'r', encoding = 'utf-8')
    file_name = 'data_testword2vecsmall.json'
    ouf = open(os.path.join(output_path, file_name), "w")

    fact = []
    print(get_batch())
    for line in inf:
        fact.append(json.loads(line)["fact"])
        if len(fact) == get_batch():
            print ("comein")
            result = solve(fact)
            cnt += len(result)
            for x in result:
                print ("ha")
                print(x)
                print(json.dumps(x), file=ouf)
            fact = []

    if len(fact) != 0:
        print(len(fact))
        print("comein")
        result = solve(fact)
        cnt += len(result)
        for x in result:
            print(json.dumps(x),file=ouf)
        fact = []
    

    ouf.close()
    print("spend time : %.9f seconds" % ((time.time()-start_time)))
