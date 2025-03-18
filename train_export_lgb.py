import pickle
from lightgbm import LGBMClassifier
import pandas as pd

from utils import read_config, map_to_category

cfg = read_config('configfiles/config.yaml')
model_params = pd.read_pickle('best_models_wo_smote')['params'][0]
# model_params['objective'] = 'multiclass'
# model_params['metric']= 'multi_logloss',

all_data = pd.read_csv(cfg['dataset'], usecols=cfg['features'] + cfg['target'])
categories = ["Toilet","Shower","Faucet","ClothesWasher","Dishwasher","Bathtub"]
# model_params['num_class'] = len(categories)
end_uses = all_data[cfg['target']].map(lambda x: map_to_category(x, categories))

# lgb_dataset = lgb.Dataset(all_data[cfg['features']], label=all_data[cfg['target']])

model = LGBMClassifier(**model_params)
model.fit(all_data[cfg['features']], end_uses)

pickle.dump(model, open('model.pkl', 'wb'))