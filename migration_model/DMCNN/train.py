import os
from config import Config


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

con = Config()
con.load_traint_data()
con.set_traint_model()
con.train()
