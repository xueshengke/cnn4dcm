% new code for new data128-c4 applied in CNN  
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
%trainData=reshape(trainData, width, height, size(trainData, 2));
%testData=reshape(testData, width, height, size(testData, 2));
trainNum = size(trainData, 3) ;
testNum = size(testData, 3) ;
clear cnn;
%% CNN 设计
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
cnn.error = [];
cnn.inputmaps = 1 ;         % gray image
opts.alpha = 1  ;
opts.batchsize = 57  ;      % needs to change according to trainNum 57 * 9 = 513
opts.numepochs = 50 ;    % long time  seconds per poches
%%
% fprintf('initiate cnn....\n');
cnn = cnnsetup(cnn, trainData, trainLabel);
%%
 %load('dcm/cnn_128_6_12_5_500_mean');  % load cnn which has been trained
%%
% fprintf('start training cnn...\n');
% tic;
cnn = cnntrain(cnn, trainData, trainLabel, opts);
[ratio, er, bad] = cnntest(cnn, testData, testLabel);
fprintf('error rate : %.6f %%\n', er );
%{
for i = 1 : 500
    cnn = cnntrain(cnn, trainData, trainLabel, opts);
    % toc;
    % fprintf('cnn training completes\n');

    % save('dcm/cnn_128_6_12_5_1000_mean', 'cnn', '-v7.3');
    % disp('model saved-->dcm/cnn_128_6_12_5_1000_mean');

    % load dcm/testData;
    % load dcm/testLabel;
    %% commence the cnn test
    % load('dcm/cnn_128_6_16_5_1000_mean'); 
    % fprintf('cnn test commences :\n');
    [ratio, er, bad] = cnntest(cnn, testData, testLabel);
    cnn.error{i} = er;
    fprintf('%d / %d error rate :%.2f \n', i, 500, er) ;

end
% fprintf('correct rate : %.2f %%\n', double(ratio * 100) );
% fprintf('wrong number : %d / %d \n', numel(bad), testNum);
fprintf('cnn end !\n');
%}

