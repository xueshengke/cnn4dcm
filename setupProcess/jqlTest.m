[dataTrain, labelsTrain, dataTest, labelsTest] = generateData_cnn();
%{
testData=reshape(dataTest, 128, 128, size(dataTest, 2));
%testData=testData-repmat(mean(testData,2), 1, size(testData,2));
[ratio, er, bad] = cnntest(cnn, testData, labelsTest);
fprintf('correct rate : %.2f %%\n', double(ratio * 100) );
%}


newData=[dataTrain,dataTest];
newLabels=[labelsTrain, labelsTest];
clear dataTrain;
clear dataTest;
clear labelsTrain;
clear labelsTest;
trainData=reshape(newData, 128, 128, size(newData, 2));
%trainData=trainData-repmat(mean(trainData,2), 1, size(trainData,2));
[ratio, er, bad] = cnntest(cnn, trainData, newLabels);
fprintf('correct rate : %.2f %%\n', double(ratio * 100) );