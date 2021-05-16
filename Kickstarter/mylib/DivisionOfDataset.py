import pandas as pd

def DivisonOfDataset(dataset, train_procent):
	train_return = dataset.loc[0:len(dataset)*(train_procent/100)]
	test_return = dataset.loc[len(dataset)*(train_procent/100):len(dataset)-1]
	
	return train_return,test_return