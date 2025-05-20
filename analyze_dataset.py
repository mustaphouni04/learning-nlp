import json
import pprint

with open("uab_summary_2024_all.json") as f:
    d = json.load(f)


#pprint.pprint(d, compact=True)

pprint.pprint(d[0], compact=True)


