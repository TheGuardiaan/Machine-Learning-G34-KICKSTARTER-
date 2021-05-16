import math
from time import time
import time as currtime
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score

def RandomSearcherFunk(model,tuning,X_train,y_train,ascoring,acv=5,aniter=20,averbose=0,job=-1):
    start = time()
    random_tuned = RandomizedSearchCV(
        model, 
        tuning, 
        n_iter=aniter, 
        random_state=42, 
        cv=acv, 
        scoring=ascoring, 
        verbose=averbose, 
        n_jobs=job, 
    )
    random_tuned.fit(X_train, y_train)
    t = time() - start
    
    return random_tuned,t

def GridSearchFunk(model,tuning,X_train,y_train,ascoring,acv=5,averbose=0,job=-1,scaler=1):

    e = SearchTimeEstimatorFunk(model,tuning, X_train,y_train,scaler,ascoring,job=job,acv=acv)
    
    start = time()
    grid_tuned = GridSearchCV(
        model,
        tuning,
        cv=acv,
        scoring=ascoring,
        verbose=averbose,
        n_jobs=job,
    )
    grid_tuned.fit(X_train, y_train)
    t = time() - start
    
    if e != None:
        scaler =scaler*0.8 + 0.2*scaler*(t/e)
    
    print(f"ended at: {currtime.localtime().tm_hour}:{currtime.localtime().tm_min}:{currtime.localtime().tm_sec}  {currtime.localtime().tm_mday}/{currtime.localtime().tm_mon}/{currtime.localtime().tm_year}")
    return grid_tuned,t,scaler

def SearchTimeEstimatorFunk(model,tuning,x,y, scaler,ascoring,acv=5, job=-1):
    tuning_parameters =tuning.copy()
    combination_count=CalcCombination(tuning_parameters)
    print(f"Datasize: {x.shape[0]}, feature: {x.shape[1]}")
    print()
    print(f"Start at: {currtime.localtime().tm_hour}:{currtime.localtime().tm_min}:{currtime.localtime().tm_sec}  {currtime.localtime().tm_mday}/{currtime.localtime().tm_mon}/{currtime.localtime().tm_year}")
    print()
    if combination_count ==1:
        return None
    
    for i in tuning_parameters:
        tuning_parameters[i]=[tuning_parameters[i][0]]
    if job <1 :
        aJob=12
    else:
        aJob=job
    
    if 'n_jobs' in model.get_params():
        
        hjob = int(np.round(aJob/5))
        joby = aJob-hjob
        
        if hjob != 0 : model.n_jobs=hjob
        else: model.n_jobs=1
   
        
    else:
        joby=job
        
    
    start = time()
    grid_tuned = GridSearchCV(
        model,
        tuning_parameters,
        cv=acv,
        scoring=ascoring,
        verbose=0,
        n_jobs=joby,
    )
    
    grid_tuned.fit(x, y)
    end = time()
    
    t = end -start
    est = t*combination_count*scaler
    currest = currtime.localtime(time() + est)
    
    esthour,estmin,estsec = SecToHourMinSec(est)
    print(f"One iteration took: {round(t,2)} secounds")
    print(f"Estimated time: {esthour}:{estmin}:{estsec}")
    print(f"Expected done at : {currest.tm_hour}:{currest.tm_min}:{currest.tm_sec} {currest.tm_mday}/{currest.tm_mon}/{currest.tm_year}")
    return est

def CalcCombination(tunning):
    comb = 1
    for i in tunning:
        comb = comb*len(tunning[i])

    print(f"Combination count: {comb}")
    return comb

def SecToHourMinSec(somesec):
    secPerMin=60
    secPerHour=secPerMin*60
    
    hour = math.floor(somesec/(secPerHour))
    hourToSec = hour*secPerHour
    
    minute = math.floor((somesec-hourToSec)/60)
    minToSec = minute*secPerMin
    sec = math.floor(somesec-(minToSec+hourToSec))
    
    return hour,minute,sec