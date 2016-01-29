function mnist_cnn_example
clear all;clc;
addpath(genpath('DeepLearnToolbox'));
load mnist_uint8;
%% reconstruct data and normalize it
train_x = double(reshape(train_x',28,28,60000))/255;
train_y = double(train_y');

test_x = double(reshape(test_x',28,28,10000))/255;
test_y = double(test_y');

%% Train a 6c-2s-12c-2s Convolutional neural network 
rand('state',0)
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
};
%% initiate cnn network
fprtinf('commence inititate cnn \n');
cnn.inputmaps = 1 ;         % gray image
cnn = cnnsetup(cnn, train_x, train_y);

opts.alpha = 1 ;
opts.batchsize = 50 ;       % select batch from train data
opts.numepochs = 10 ;
%% start training cnn network
fprintf('commence training cnn \n');
tic ;
cnn = cnntrain(cnn, train_x, train_y, opts);
toc ;
%% start test cnn network
fprintf('commence testing cnn \n');
[ratio, er, bad] = cnntest(cnn, test_x, test_y);
fprintf('Accuracy %.2f %%\n', ratio * 100) ;
% plot mean squared error
figure; plot(cnn.rL);

assert(er<0.12, 'Too big error');
