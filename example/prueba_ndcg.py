
import os
import sys

import numpy as np
import pandas as pd

import config
import metrics


ndcg = []
map10 = []
novelty = []
id = []
df_games = pd.read_csv(config.DATA_GAME)



id.append("DeepFM")
dfResult = pd.read_csv(config.SUB_DIR + "/DeepFM_" + config.OUTPUT_NAME)

map10.append(metrics.mean_average_precision_total(dfResult))
print(map10)

id.append("FM")
dfResult = pd.read_csv(config.SUB_DIR + "/FM_" + config.OUTPUT_NAME)

map10.append(metrics.mean_average_precision_total(dfResult))
print(map10)

id.append("DNN")
dfResult = pd.read_csv(config.SUB_DIR + "/DNN_" + config.OUTPUT_NAME)

map10.append(metrics.mean_average_precision_total(dfResult))
print(map10)

pd.DataFrame({"Modelos": id, "MAP": map10}).to_csv(
        os.path.join(config.SUB_DIR, "map10.csv"), index=False, float_format="%.5f")

sys.exit()








id.append("DeepFM")
dfResult = pd.read_csv(config.SUB_DIR + "/DeepFM_" + config.OUTPUT_NAME)

map10.append(metrics.mean_average_precision_total(dfResult))
print(map10)

ndcg.append(metrics.ndcg_total_at_k(dfResult))
print(ndcg)


novelty.append(metrics.novelty_total(dfResult, df_games))
print(novelty)





id.append("FM")
dfResult = pd.read_csv(config.SUB_DIR + "/FM_" + config.OUTPUT_NAME)

ndcg.append(metrics.ndcg_total_at_k(dfResult))
print(ndcg)

map10.append(metrics.mean_average_precision_total(dfResult))
print(map10)

novelty.append(metrics.novelty_total(dfResult, df_games))
print(novelty)



id.append("DNN")
dfResult = pd.read_csv(config.SUB_DIR + "/DNN_" + config.OUTPUT_NAME)

ndcg.append(metrics.ndcg_total_at_k(dfResult))
print(ndcg)

map10.append(metrics.mean_average_precision_total(dfResult))
print(map10)

novelty.append(metrics.novelty_total(dfResult, df_games))
print(novelty)


pd.DataFrame({"Modelos": id, "NDCG": ndcg, "MAP": map10, "Novelty": novelty}).to_csv(
        os.path.join(config.SUB_DIR, "metrics.csv"), index=False, float_format="%.5f")

