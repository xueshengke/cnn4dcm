% new code for new data128-c4 applied in CNN  
clear all;  clc;
addpath(genpath('DeepLearnToolbox'));
%% load data from jpg or file system
width=128;
height=128;
%% load dataset
[trainData, trainLabel, testData, testLabel] = generateData_cnn();
%% if data have exist as mat in file system, just load
% load dcm/trainData;
% load dcm/trainLabel;
% end
% batch mean to zero
% trainData=trainData-repmat(mean(trainData,2),1,size(trainData,2));
% testData=testData-repmat(mean(testData,2),1,size(testData,2));

trainData=reshape(trainData, width, height, size(trainData, 2));
testData=reshape(testData, width, height, size(testData, 2));
trainNum = size(trainData, 3) ;
testNum = size(testData, 3) ;
%% CNN design
rand('state',0)
cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 3)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 5, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
%  struct('type', 'c', 'outputmaps', 3, 'kernelsize', 7)
%  struct('type', 's', 'scale', 3)
    };
cnn.inputmaps = 1 ;         % gray image
opts.alpha = 1  ;
opts.batchsize = 57  ;      % needs to change according to trainNum 57 * 9 = 513
opts.numepochs = 100 ;    % long time  seconds per poches
%%
% fprintf('initiate cnn....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);
%%
 %load('dcm/cnn_128_6_12_5_500_mean');  % load cnn which has been trained
%%
fprintf('start training cnn...\n');
tic;
cnn = cnntrain(cnn, trainData, trainLabel, opts);
toc;
fprintf('cnn training completes\n');

% save('dcm/cnn128_6_16_5_100_j_mean', 'cnn', '-v7.3');
% disp('model saved-->dcm/cnn128_6_16_5_100_j_mean');

% load dcm/testData;
% load dcm/testLabel;
%% commence the cnn test
% load('dcm/cnn_128_6_16_5_1000_mean'); 
fprintf('cnn test commences :\n');
[ratio, er, bad] = cnntest(cnn, testData, testLabel);
fprintf('correct rate : %.2f %%\n', double(ratio * 100) );
fprintf('wrong number : %d / %d \n', numel(bad), testNum);
fprintf('cnn end !\n');


