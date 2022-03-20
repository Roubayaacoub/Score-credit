
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from flask import Flask, jsonify, request
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app)

#pred_model_banq2 = pickle.load( open( "pred_model_banq2.md", "rb" ) )
data_final = pickle.load( open( "data_final.p", "rb" ) )


@app.route('/')
def home():
    return 'Entrer une ID client dans la barre URL'

@app.route('/<int:ID>/')
def requet_ID(ID):
    
    if ID not in list(data_final['SK_ID_CURR']):
        result = 'Ce client n\'est pas dans la base de donnée'
    else:
        data_final_ID=data_final[data_final['SK_ID_CURR']==int(ID)]

   
        if y_Target == 0:
           result=('ce client est solvable avec un taux de risque de '+ str(np.around(data_final.predictions*100,2))+'%')

        elif y_Target == 1:
            result=('ce client est non solvable avec un taux de risque de '+ str(np.around(data_final.predictions*100,2))+'%')
  
    return result


if __name__ == '__main__':
  app.run()