from sklearn.preprocessing import OneHotEncoder as OHEncoder
import pandas as pd

def EncodeColToBinary(dataset,col):

    onencoder = OHEncoder()

    cat_fac_encoded,cat_fac_decoder = pd.factorize(dataset[col])
    #dataset[col+"_encoded"] = cat_fac_encoded

    cat_fac_encoded_vector=cat_fac_encoded.reshape(len(cat_fac_encoded),1)
    cat_fac_encode_matrix = onencoder.fit_transform(cat_fac_encoded_vector)

    for i in range(len(cat_fac_decoder)):
        dataset[col+"_"+cat_fac_decoder[i]] = cat_fac_encode_matrix.toarray()[:,i].astype(int)
    
    return dataset