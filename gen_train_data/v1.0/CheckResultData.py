
from global_variables import *
import numpy as np


#%%
def readAudioDataFrom_txt():
    returnList = fpro.find_data_files(txtFilesPath, ".txt")
    print(returnList)
    return returnList


#%%
def processTextData(inputList):

    returnList = list()
    for one in inputList:
        
        if "[" in one:
            one = one[1:]    
        if "]" in one:
            one = one[:-1]
        if "," in one:
            one = one[:-1]

        tempFloat = float(one)*1000
        tempInt = int(tempFloat)
        returnList.append(tempInt)
    
    print(len(returnList))
    returnList = np.array(returnList, dtype=np.int16)
    
    return returnList


#%%
def readFilesData(inputList):
    
    resultList = list()

    for oneFile in inputList:
        with open(oneFile, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                line = line.split()
                if not line:break
           
                returnData = processTextData(line)
                resultList.append(returnData)
                
    return resultList


#%%
def genWavFiles(inputList):

    for i, oneFileData in enumerate(inputList):
        filename = txtFilesPath+"\\"+"example0"+str(i)+".wav"
        wavfile.write(filename, 16000, oneFileData)
    
    return


#%%
def main():
    returnList = readAudioDataFrom_txt()
    returnList = readFilesData(returnList)
    genWavFiles(returnList)
    




#%%
if __name__ == "__main__":
    print('hello, world~!!')
    main()
