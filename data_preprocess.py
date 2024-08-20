from config import config
import pandas

data_dir = config['dynamic']
csv_path = data_dir + '/FileList.csv'

with open(csv_path) as f:
    data = pandas.read_csv(f)

