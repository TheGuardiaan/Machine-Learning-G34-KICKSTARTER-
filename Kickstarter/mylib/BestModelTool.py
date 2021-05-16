from mylib.L8ReportTool import FullReport
from datetime import datetime

def CheckNewBest(model,finalEst,filename,writeFile,modelType):
	print(f"New Model score: {model.best_score_}")
	if modelType != None:
		newModelType = modelType
	else:
		newModelType = GetEstType(model)
	
	oldEst = finalEst.get(newModelType)
	if oldEst != None:
		print(f"Old Model score: {oldEst.best_score_}")

	if oldEst== None or model.best_score_ > oldEst.best_score_:
		finalEst[newModelType]=model
		if writeFile:
			WriteLogFile(model,filename)
		print("New model saved")
			
	return finalEst.get(newModelType)

def GetEstType(model):
	return type(model.estimator).__name__

def PrintHighScore(finalArr):
	for i in finalArr:
		print(f"{i} : {finalArr[i].best_score_}") #GetEstType(finalArr[i])
		
def HandleNewBest(model,x,y,t,finalModel,report=True,filename="logfile.txt",writeFile=False,modelType=None):
	if model != None:
		# Report result
		CheckNewBest(model,finalModel,filename,writeFile,modelType)
		if report:
			print()
			PrintHighScore(finalModel)
			print()
			return FullReport(model, x, y, t)

	return None,None
	
def WriteLogFile(model,filename):
	score = model.best_score_
	model = model.best_estimator_
	now_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")
	with open(filename,'a') as f:
		f.write(f"{now_time}\n\tScore: {score}\n\tModel: {model}\n")