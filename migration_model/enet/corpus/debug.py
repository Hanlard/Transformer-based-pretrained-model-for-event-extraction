import json
f=open('C:\Hanlard\\NLP\模型\事件抽取\EMNLP2018-JMEE-master\\ace-05-splits\sample.json',encoding="utf-8")
e=json.load(f)
event_json=e[0]
from enet.corpus.Sentence import *
sentence=Sentence(json_content=event_json)
generateAdjMatrix=sentence.generateAdjMatrix()