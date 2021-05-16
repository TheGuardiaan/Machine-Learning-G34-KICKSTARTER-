def rfc_fitting(minImp) :
    return RFC(n_estimators=35,
        criterion='entropy', 
        max_depth=None, 
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=minImp,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=10,
        random_state=42,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None)

def Train(X_train, y_train, X_val, y_val, n_epochs, verbose=False):
    print("Training...n_epochs=",n_epochs)
    
    train_errors, val_errors = [], []
    
    minImpArr = np.logspace(-1,-8,n_epochs)
    
    for epoch in range(n_epochs):
        rfc = None
        rfc = rfc_fitting(minImpArr[epoch])
        rfc.fit(X_train, y_train)
        
        y_train_predict = rfc.predict(X_train)
        y_val_predict   = rfc.predict(X_val)
        
        f1_train=f1(y_train, y_train_predict, average='micro')#, squared=False)
        f1_val  =f1(y_val  , y_val_predict,average='micro')#, squared=False)

        train_errors.append(f1_train)
        val_errors  .append(f1_val)
        if verbose:
            print(f"  epoch={epoch:4d}, f1_train={f1_train:4.2f}, f1_test={f1_val:4.2f}")

    return train_errors, val_errors
    
X_train_fitting,X_test_fitting,Y_train_fitting,Y_test_fitting = DivideData(X_data,Y_data,30)

n_epochs = 100
train_errors, val_errors = Train(X_train_fitting, Y_train_fitting['outcome'], X_test_fitting, Y_test_fitting['outcome'], n_epochs, True)


plt.figure(figsize=(10,5))

plt.plot(train_errors, "b--", linewidth=2, label="Training set")
plt.plot(val_errors, "g-", linewidth=3, label="Test set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("f1", fontsize=14)
plt.show()