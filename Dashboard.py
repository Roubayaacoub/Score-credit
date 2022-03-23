# streamlit run C:\Users\Rouba\Downloads\python\API.py
import streamlit as st
import lime
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as pl
import seaborn as sn
import time
import lime.lime_tabular



#Chargement des dataset
data_final = pickle.load( open( "data_final.p", "rb" ) )
lime_data = pickle.load( open( "lime_data.p", "rb" ) )

#data_info = pickle.load( open( "data_info.p", "rb" ) )
data_train = pickle.load( open( "data_train_final.p", "rb" ) )
data_test = pickle.load( open( "data_test_final.p", "rb" ) )
data_train['TARGET'] = data_train['TARGET'].replace({True:0,False:1})

clf_final = pickle.load( open( "model.p", "rb" ) )

new_train = pd.read_csv('application_train.csv')
#new_train = pd.read_csv( "new_train.csv" ) 

data_train_drop = data_final[data_final['probabilities'].isna()]
data_test_drop = data_final[~data_final['probabilities'].isna()]



data_train_drop['probabilities'] = data_train_drop['probabilities'].fillna(1)
data_final['probabilities'] = data_final['probabilities'].fillna(1)


#fenetre input
st.title('Scoring client pour crédit')
st.subheader("Prédictions du score bancaire du client")
id_input = st.text_input('Merci de saisir l\'identifiant du client:', )
st.write('Exemple d\'ID client: 171775 ; 28005; 445016')

@st.cache
def requet_ID(ID):
    
    ID_client=int(ID)
    
    if ID_client not in list(data_final['SK_ID_CURR']) :
        result = 'Ce client n\'est pas dans la base de données.'
    
    else:
        
        #Récupération des infos clients
        X_ID = data_final[data_final['SK_ID_CURR']==ID_client].copy()
        X = X_ID.drop(['SK_ID_CURR', 'TARGET'],axis=1)
        y_ID = X_ID['SK_ID_CURR']
        y_Target = X_ID['TARGET']
        
        y_prob = X_ID['probabilities']
      
        if y_Target.item() == 0:
          result=('Ce client n\'a pas des difficultés de paiement avec un taux de risque de '+ str(round(y_prob.iloc[0]*100,2))+'%')
        
        elif y_Target.item() == 1:
          result=('Ce client a des difficultés de paiement avec un taux de risque de '+ str(round(y_prob.iloc[0]*100,2))+'%')
      
    return result


def profil_client(ID,lime_data):
  
    lime_data['AGE'] = round(abs(lime_data['DAYS_BIRTH']/365)).astype(int)
    ID_c = int(ID)  
    #INFO CLIENT original
    info_client_T = lime_data[lime_data['SK_ID_CURR']==ID_c]
    enfant_client = info_client_T['CNT_CHILDREN'].item()
    age_client = info_client_T['AGE'].item()
    genre_client = info_client_T['CODE_GENDER'].item()
    region_client = info_client_T['REGION_RATING_CLIENT'].item()
    
    ## create an array 
    arr_cl = []
    
    arr_cl.append(age_client)
    arr_cl.append(genre_client)
    arr_cl.append(enfant_client)
    arr_cl.append(region_client)
    
    ## create a dataframe with the original entry info about a client
    data_client = pd.DataFrame(arr_cl)
    data_client = data_client.T
    data_client.columns=['AGE','GENRE','ENFANT','CODE_REGION']
    data_client.index=[str(ID_c)]
    
    return data_client

def comparaison_indiv(data_train, data_test,id_input):
    
    
    ID = data_test[data_test['SK_ID_CURR'] == int(id_input)].index.values
    data_train_drop = data_train.drop(columns=['TARGET','SK_ID_CURR'])
    data_test_drop = data_test.drop(columns=['SK_ID_CURR'])
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data = np.array(data_train_drop), 
                                                   feature_names=data_train_drop.columns,discretize_continuous=False,
                                                  verbose=True, mode='classification')
    exp = explainer.explain_instance(data_test_drop.iloc[ID[0]], clf_final.predict_proba)
    
    components.html(exp.as_html(), height=800)




def st_lime(plot, height=None):
    lime_html = f"<head>{lime.getjs()}</head><body>{plot.html()}</body>"
    components.html(lime_html, height=height)


def plot_ft_global(ID,lime_data,ft_voisin):
    
    ID = int(ID)  
    lime_data_id = lime_data[lime_data['SK_ID_CURR']==ID]
    
    if ft_voisin == True:
        lime_data = lime_data.drop(index=lime_data_id.index)
        lime_data = lime_data.reset_index()
        lime_data = lime_data.drop(columns=['index'])
        
        
        palette = sn.color_palette('hls')
        select_features=['AMT_INCOME_TOTAL','AMT_GOODS_PRICE','EXT_SOURCE_3', 'EXT_SOURCE_2','EXT_SOURCE_1']

        color0 = palette[0]
        color1 = palette[1]


        for j in select_features:
            print(j)
            #pl.figure(figsize=(8,5))
            x = lime_data[lime_data['TARGET']==0][j]
            y = lime_data[lime_data['TARGET']==1][j]
            
            fig,ax = pl.subplots( figsize=(10,4))
            pl.hist([x,y],label = ['0','1'], color = [color0,color1],bins=7)



            pl.legend()
            pl.axvline(lime_data_id[j].item(),linewidth=4, color='black', linestyle="--")
            pl.xlabel(j)
            pl.ylabel('Nb de client')
        
            
            st.pyplot(fig)
            
    if ft_voisin == False:
        
       select_features=['CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_GOODS_PRICE','EXT_SOURCE_3', 'EXT_SOURCE_2','EXT_SOURCE_1','FLAG_OWN_CAR_N']
       data = pd.DataFrame()
       for j in select_features:
            array_feature = [lime_data[j].mean(),lime_data[lime_data['TARGET']==0][j].mean(), lime_data[lime_data['TARGET']==1][j].mean(), lime_data_id[j].item()]
            data_feature = {'valeur_moyenne':array_feature[0],
                        'Target_0':array_feature[1],
                        'Target_1':array_feature[2],
                        'ID_'+str(ID):array_feature[3]}
            data = pd.DataFrame(data_feature, index=[j])
            print('Comparaison des infos du client pour differents features',j)

            fig,ax = pl.subplots(figsize=(10,4))
            sn.barplot(data = data)
            #sn.despine(bottom = True, left = True)
            sn.despine()
            pl.xlabel(j)
            pl.ylabel('moyenne de nbre des clients')
            
            st.pyplot(fig)


def comparaison_client_voisin(ID,lime_data,new_train):
    lime_data['AGE']=round(abs(lime_data['DAYS_BIRTH']/365)).astype(int)
    new_train['AGE']=round(abs(new_train['DAYS_BIRTH']/365)).astype(int)
  
    ID_c=int(ID)

    #INFO CLIENT lime
    data_id = lime_data[lime_data['SK_ID_CURR']==ID_c]
    enfant_id = data_id['CNT_CHILDREN'].item()
    age_id = data_id['AGE'].item()
    genre_id = data_id['CODE_GENDER'].item()
    region_id = data_id['REGION_RATING_CLIENT'].item()
    status_id = data_id['NAME_FAMILY_STATUS'].item()

    #PROCHE VOISIN
    proche_voisin = new_train[(new_train['AGE']==age_id) &
                          (new_train['CODE_GENDER']==genre_id)& 
                          (new_train['CNT_CHILDREN']==enfant_id) &
                          (new_train['REGION_RATING_CLIENT']==region_id) &
                          (new_train['NAME_FAMILY_STATUS']==status_id) ]

    if len(proche_voisin) < 15:
         set_client_voisin = proche_voisin.sample(len(proche_voisin),random_state=42)
    if len(proche_voisin) >= 15:
        set_client_voisin = proche_voisin.sample(15,random_state=42)
    
    set_client_voisin = pd.concat([set_client_voisin, data_id])
    plot_ft_global(ID_c, set_client_voisin, ft_voisin=True)
    


#######


if id_input == '':
    st.write('Veuillez entrer un ID')
    
else:
    r_ID=requet_ID(id_input)
    st.write(r_ID)
    
    if r_ID != 'Ce client n\'est pas dans la base de données.':
    
        option = st.sidebar.selectbox('Interprétation',['','Globale','Individuelle','Profils similaires'])  
         
        
        if option == 'Globale':
            st.sidebar.subheader('Profil client '+ str(id_input))
            st.sidebar.write(profil_client(id_input,lime_data))
            st.write(plot_ft_global(id_input,lime_data, ft_voisin=False))
            
        elif option == 'Individuelle':
            st.sidebar.subheader('Profil client '+ str(id_input))
            st.sidebar.write(profil_client(id_input,lime_data))
            st.write(comparaison_indiv(data_train, data_test, id_input))

           
  
        elif option == 'Profils similaires':
            st.sidebar.subheader('Profil client '+ str(id_input))
            st.sidebar.write(profil_client(id_input,lime_data))
            st.write(comparaison_client_voisin(id_input,lime_data,new_train))
        else:
            st.sidebar.write('Chosissez un mode de comparaison')

