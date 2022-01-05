import csv
import numpy as np


    
    #line_count = 0
train_set = np.array([])

sample_length=100

for offset in range(0,sample_length):
    sample = []
    line_count=0
    with open('C:\\Users\\Andrew\\Documents\\StockML\\GIS.csv') as csv_file:
        #print(offset)
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            #print('test1')
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            elif line_count <= offset:
                line_count+=1
                continue
            else: 
                if((line_count-offset)%(sample_length+1) == 0 and line_count>offset):
                    #print('line_count: '+str(line_count))
                    #print('offset: ' + str(offset))
                    if(float(row[1])>= sample[-1]):
                        sample.append(1)
                    else:   
                        sample.append(0)
                    
                    if(len(train_set)==0):
                        train_set = np.append(train_set, sample)
                    else:
                        #print(train_set[-1])
                        #print(sample)
                        train_set = np.vstack([train_set, sample])
                    sample = []
                else:
                    sample.append(float(row[1]))
                line_count+=1


#print(train_set)
np.save('GIS_train_set', train_set)

raw_data = np.load('GIS_train_set.npy')
x=np.array([i[0:-1] for i in raw_data])
y=np.array([])
for i in raw_data:
    if len(y)==0:
        if(i[-1]==1):
            y=np.append(y, [1,0])
        else:
            y=np.append(y, [0,1])
    else:
        if(i[-1]==1):
            y=np.vstack([y, [1,0]])
        else:
            y=np.vstack([y, [0,1]])
#y=np.array([[i[-1],0] for i in raw_data if i[-1]==0])
print(len(x))
print(len(y))