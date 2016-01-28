clear all;close all;clc;
addpath(genpath('DeepLearnToolbox'));
%% load data from jpg or file system
width=384;
height=384;
if ~exist('dcm/trainData.mat', 'file') || ~exist('dcm/testData.mat', 'file') || ~exist('dcm/trainLabel.mat', 'file') || ~exist('dcm/testLabel.mat', 'file')
    path{1}='/home/xueshengke/data384/bz';
    path{2}='/home/xueshengke/data384/norm';
    %% load dataset
    dataset = loadImageData(path, width, height);
    %% generate train/test data
    trainNum=350;
    testNum=29;
    randNum=randperm(size(dataset{1}.img, 2));
    trainOrder1=randNum(1:trainNum);
    testOrder1=randNum(trainNum+1:trainNum+testNum);
    randNum=randperm(size(dataset{2}.img, 2));
    trainOrder2=randNum(1:trainNum);
    testOrder2=randNum(trainNum+1:trainNum+testNum);
    trainData=[dataset{1}.img(:, trainOrder1), dataset{2}.img(:,trainOrder2)];
    testData=[dataset{1}.img(:, testOrder1), dataset{2}.img(:, testOrder2)];
    la1=[1,0]';
    la2=[0,1]';
    trainLabel=[repmat(la1, 1, trainNum), repmat(la2,1, trainNum)];
    testLabel=[repmat(la1, 1, testNum), repmat(la2, 1, testNum)];
    %% train data shuffle
    randNum=randperm(size(trainData, 2));
    trainData=trainData(:, randNum);
    save('dcm/trainData', 'trainData', '-v7.3');
    save('dcm/trainLabel', 'trainLabel', '-v7.3');
    save('dcm/testData', 'testData', '-v7.3');
    save('dcm/testLabel', 'testLabel', '-v7.3');
    disp('train/test data have saved in folder dcm/.');
    clear dataset;
else
    %% if data have exist as mat in file system, just load
    load dcm/trainData;
    load dcm/trainLabel;
end
trainData=reshape(trainData, width, height, size(trainData, 2));
%% CNN 设计
rand('state',0)
cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 3)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 5, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 7)
    struct('type', 's', 'scale', 3)
    };
cnn.inputmaps = 1 ;
opts.alpha = 1  ;
opts.batchsize = 35  ;
opts.numepochs = 50 ; % long time 152 seconds per poches
fprintf('初始化cn网络....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);
fprintf('开始训练cnn网络...\n');tic;
cnn = cnntrain(cnn, trainData, trainLabel, opts);toc;
fprintf('cnn训练结束\n');

save('dcm/cnn', 'cnn', '-v7.3');
disp('模型已经保存-->dcm/cnn');

load dcm/testData;
load dcm/testLabel;
testData=reshape(testData, width, height, size(testData, 2));
fprintf('cnn开始测试\n');
[ratio, er, bad] = cnntest(cnn, testData, testLabel);
fprintf('correct : %.2f %%\n', double(ratio * 100) );

%{
otherxjjData = otherxjjData( randperm(otherxjj) , :) ;
randomDataIndex = randperm(xjj + other) ;
%% get train data and labels
xjjTrainData = zeros(xjj + other, imageSize * imageSize) ;
xjjTrainData(1 : xjj, :) = xjjData(: , :) ;
xjjTrainData(xjj + 1 : xjj + other, :) = otherxjjData(1 : other, :);
xjjTrainData = xjjTrainData( randomDataIndex , : ) ;

xjjTrainLabel = zeros(2 , xjj + other) ;
xjjTrainLabel(:, 1 : xjj) = repmat([1, 0]' , 1 , xjj );
xjjTrainLabel(:, xjj + 1 : xjj + other) = repmat([0, 1]' , 1 , other) ;
xjjTrainLabel = xjjTrainLabel(:, randomDataIndex) ;

% reshape data  into size * size * channel * samples, and normalization to 0~1 for each pixel
 xjjTrainData = double(reshape(xjjTrainData', imageSize, imageSize, channel * (xjj + other) ) )/255 ;
 
%% get test data and labels from train data and other data except those in otherxjjData
randomxjjDataIndex = randperm(xjj) ;
xjjTestData = zeros(xjjTestNum + otherTestNum, imageSize * imageSize) ;
xjjTestData(1 : xjjTestNum, :) = xjjData(randomxjjDataIndex(1 : xjjTestNum) , :) ;
%% select data from otherxjjData excluding those in train
xjjTestData(xjjTestNum + 1 : xjjTestNum + otherTestNum, :) = otherxjjData(other + 1 : other + otherTestNum, :) ;
%% select data from otherxjjData in those in train
%xjjTestData(xjjTestNum + 1 : xjjTestNum + otherTestNum, :) = otherxjjData(1 : otherTestNum, :) ;
randomTestIndex = randperm(xjjTestNum + otherTestNum) ; 
xjjTestData = xjjTestData(randomTestIndex, :) ;

xjjTestLabel = zeros(2, xjjTestNum + otherTestNum) ;
xjjTestLabel(:, 1 : xjjTestNum) = repmat([1, 0]' , 1 , xjjTestNum) ;
xjjTestLabel(:, xjjTestNum + 1 : xjjTestNum + otherTestNum) = repmat([0, 1]' , 1 , otherTestNum) ;
xjjTestLabel = xjjTestLabel(:, randomTestIndex) ;

% reshape data  into size * size * channel * samples, and normalization to 0~1 for each pixel
 xjjTestData = double(reshape(xjjTestData', imageSize, imageSize, channel * (xjjTestNum + otherTestNum) ) )/255 ;
 
%%
trainSamples=size(xjjTrainData,1);
fprintf('----------------------------------------------\n');
fprintf('train data number: %d\n', trainSamples);
fprintf('channel: %d\n',channel);
fprintf('test data number: %d\n', xjjTestNum + otherTestNum);
fprintf('image size: %d\n', imageSize );
fprintf('----------------------------------------------\n\n');

%% 计算均值
% xjjTrainData=reshape(xjjTrainData, trainSamples, channel, 512 * 512 );   %512*512
% if DEBUG_IN_CENTOS
% 	fprintf('formData size: \n');
% 	size(xjjTrainData)
% end
% meanValue=mean(xjjTrainData, 3);
% if DEBUG_IN_CENTOS
% 	fprintf('meanValue size:\n');
% 	size(meanValue)
% end
% meanValue=repmat(meanValue, [1 1 512 * 512] );      %512*512
% xjjTrainData=xjjTrainData - meanValue;                        %均值为0的数据
% xjjTrainData=reshape( xjjTrainData, trainSamples, channel * 512 * 512 );     %512*512

% testSet=zeros(4,size(formData, 2));
% trainSet=zeros(m-4, size(formData,2));
% a=1;b=numDjj;
% x=randperm(b-a+1)+a-1;
% num1=x(1);
% num2=x(2);
% a=1+numDjj;b=m;
% x=randperm(b-a+1)+a-1;
% num3=x(1);
% num4=x(4);
% fprintf('产生的4个随机数：%d\t%d\t%d\t%d\n',num1, num2,num3,num4);
% testSet(1,:)=formData(num1,:);
% testSet(2,:)=formData(num2,:);
% testSet(3,:)=formData(num3,:);
% testSet(4,:)=formData(num4,:);
% k=1;
% for i=1:numDjj
% 	if i ~= num1 && i~=num2
% 		trainSet(k,:)=formData(i,:);
% 		k=k+1;
% 	end
% end
% k=numDjj-2+1;
% for i=(numDjj+1):m
% 	if i~=num3 && i~=num4
% 		trainSet(k,:)=formData(i,:);
% 		k=k+1;
% 	end
% end
% trainSet = double(reshape(trainSet',512,512,channel,m-4))/255;
% testSet = double(reshape(testSet', 512, 512, channel, 4))/255;
% lab111=[1 0]';
% lab222=[0 1]';
% trainLabel=[lab111, lab111, lab111, lab111, lab111];
% trainLabel=[trainLabel, lab111, lab111, lab111, lab111, lab111];
% trainLabel=[trainLabel, lab222, lab222, lab222, lab222, lab222];
% trainLabel=[trainLabel, lab222, lab222, lab222, lab222, lab222];
% trainLabel=[trainLabel, lab222, lab222, lab222, lab222];
% testLabel=[lab111, lab111, lab222, lab222];
% 
% 
% ri = randperm(m-4);
% for i=1:(m-4)
%     newTrainIn(:,:,:,i)=trainSet(:,:,:,ri(i));
%     newTrainLab(:,i)=trainLabel(:,ri(i));
% end
% trainSet=newTrainIn;
% trainLabel=newTrainLab;
% clear newTrainIn;
% 
% ri=randperm(4);
% for i=1:4
%     newTestIn(:,:,:,i)=testSet(:,:,:,ri(i));
%     newTestLab(:,i)=testLabel(:,ri(i));
% end
% testSet=newTestIn;
% testLabel=newTestLab;
% clear newTestIn;

% trainLabel
% testLabel


%% CNN 设计
rand('state',0)
cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 5, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 10, 'kernelsize', 3)
    struct('type', 's', 'scale', 2)
		struct('type', 'c', 'outputmaps', 6, 'kernelsize', 3)
		struct('type', 's', 'scale', 2)
		struct('type', 'c', 'outputmaps', 3, 'kernelsize', 5)
		struct('type', 's', 'scale', 2)
};
cnn.inputmaps = channel ;
opts.alpha = 1  ;
opts.batchsize = 20  ;
opts.numepochs = 100 ; % long time 152 seconds per poches
fprintf('初始化cn网络....\n');
%cnn = cnnsetup(cnn, trainSet, trainLabel);
cnn = cnnsetup(cnn, xjjTrainData, xjjTrainLabel);
tic;
fprintf('开始训练cnn网络...\n');
cnn = cnntrain(cnn, xjjTrainData, xjjTrainLabel, opts);
toc;
fprintf('cnn训练结束\n');
save('matdat/cnn', 'cnn', '-v7.3');
disp('模型已经保存-->matdat/cnn');
%%
fprintf('cnn开始测试\n');
[ratio, er, bad] = cnntest(cnn, xjjTestData, xjjTestLabel);
fprintf('correct : %.2f %%\n', double(ratio * 100) );
%ratio = ratio * 100;
%disp(ratio);
fprintf('error : %.2f %%\n', double(er * 100) );
fprintf('---------Finished !------------\n' );
%er = er * 100;
%disp(er);
% disp('错误数量:');
% disp(bad);
%}













