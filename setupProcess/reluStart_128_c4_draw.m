% new code for new data128-c4 applied in CNN   activation function = relu
%clear all; close all; clc;
clc;
addpath(genpath('DeepLearnToolbox'));
%% load data from jpg or file system
width=128;
height=128;
%% load dataset
%[trainData, trainLabel, testData, testLabel] = generateData_cnn();
%% if data have exist as mat in file system, just load
% load dcm/trainData;
% load dcm/trainLabel;
% end
% trainData = trainData - repmat(mean(trainData,2),1, size(trainData,2));
%trainData=reshape(trainData, width, height, size(trainData, 2));
% testData = testData - repmat(mean(testData,2),1, size(testData,2));
%testData=reshape(testData, width, height, size(testData, 2));
trainNum = size(trainData, 3) ;
testNum = size(testData, 3) ;
%% CNN 设计
clear cnn;
rand('state',0)
cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 16, 'kernelsize', 3)
    struct('type', 's', 'scale', 2)
    struct('type', 'c', 'outputmaps', 5, 'kernelsize', 5)
    struct('type', 's', 'scale', 2)
%  struct('type', 'c', 'outputmaps', 3, 'kernelsize', 7)
%  struct('type', 's', 'scale', 3)
    };
cnn.error = [] ;
cnn.inputmaps = 1 ;         % gray image
opts.alpha = 1  ;
opts.batchsize = 57  ;      % needs to change according to trainNum 57 * 9 = 513
opts.numepochs = 58 ;      % long time  seconds per poches
 opts.minerror = 1e-6;
%%
 cnn = cnnsetup(cnn, trainData, trainLabel);
%%
% load('dcm/relu_cnn_128_3_6_5_500_mean');  % load cnn which has been trained
%%
% fprintf('start training reluCnn...\n');
% tic;

    
    fprintf('^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^\n');
    cnn = relucnntrain(cnn, trainData, trainLabel, opts);
    [~, er, ~] = relucnntest(cnn, testData, testLabel) ;
    fprintf('error rate %.6f \n', er) ;
    disp('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^');
%{
for k = 1 : 10
cnn = relucnntrain(cnn, trainData, trainLabel, opts);

% toc;
% fprintf('reluCnn training finished. \n');

% save('dcm/relu_cnn_128_6_16_5_500_mean', 'cnn', '-v7.3');
 %disp('model saved-->dcm/relu_cnn_128_6_16_5_500_mean');

% load dcm/testData;
% load dcm/testLabel;
%% commence the cnn test
% fprintf('cnn test commences :\n');



% fprintf('correct rate : %.2f %%\n', double(ratio * 100) );
% fprintf('wrong number : %d / %d \n', numel(bad), testNum);
end
[ratio, er, bad] = relucnntest(cnn, testData, testLabel) ;
cnn.error(k) = er ;
fprintf('%d / %d error rate %.4f \n',k ,500, er) ;
% [ratio, er, bad] = relucnntest(cnn, testData, testLabel) ;
% 
%  fprintf('correct rate : %.2f %%\n', double(ratio * 100) );
%  fprintf('wrong number : %d / %d \n', numel(bad), testNum);

fprintf('cnn end !\n');
    %}


