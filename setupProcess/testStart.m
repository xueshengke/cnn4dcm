clear all;
clc;
DEBUG_IN_CENTOS=1;
addpath(genpath('DeepLearnToolbox'));
load formData
load labelData
load params
m=size(formData,1);
fprintf('----------------------------------------------\n');
fprintf('样本数量: %d\n', m);
fprintf('选取的dicom切片数：%d\n',channel);
fprintf('大结节的dicom份数：%d\n', numDjj);
fprintf('小结节的dicom份数：%d\n',numXjj);
fprintf('----------------------------------------------\n\n');

%% 计算均值
formData=reshape(formData, m, channel, 512*512); 
if DEBUG_IN_CENTOS
	fprintf('formData size: \n');
	size(formData)
end
meanValue=mean(formData,3);
if DEBUG_IN_CENTOS
	fprintf('meanValue size:\n');
	size(meanValue)
end
meanValue=repmat(meanValue,[1 1 512*512]);
formData=formData-meanValue;%均值为0的数据
formData=reshape(formData, m, channel*512*512);
trainSet = double(reshape(formData',512,512,channel,m))/255;
lab111=[1 0]';
lab222=[0 1]';
trainLabel=[lab111, lab111, lab111, lab111, lab111];
trainLabel=[trainLabel, lab111, lab111, lab111, lab111, lab111,lab111,lab111];
trainLabel=[trainLabel, lab222, lab222, lab222, lab222, lab222,lab222,lab222];
trainLabel=[trainLabel, lab222, lab222, lab222, lab222, lab222];
trainLabel=[trainLabel, lab222, lab222, lab222, lab222];

disp('trainSet size');
disp(size(trainSet,4));
disp('trainLabel size');
disp(trainLabel);
disp('trainLabel');

disp('load model');
load matdat/cnn;
disp('test');
[er, bad] = cnntest(cnn, trainSet, trainLabel);

disp('the error:');
er = er * 100
%disp(er);
disp('error index:');
disp(bad)










