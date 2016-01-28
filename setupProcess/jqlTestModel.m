%load cnn;
[~, ~, testData, testLabel] = generateData_cnn();
%testData = testData - repmat(mean(testData,2),1, size(testData,2));
testData=reshape(testData, width, height, size(testData, 2));

%[ratio, er, bad] = relucnntest(cnn, testData, testLabel);
[ratio, er, bad] = cnntest(cnn, testData, testLabel);
fprintf('correct rate : %.2f %%\n', double(ratio * 100) );
fprintf('wrong number : %d / %d \n', numel(bad), testNum);
fprintf('cnn end !\n');

 net = cnnff(cnn, testData);
 values=net.o;
 values=exp(values);
 values=values./repmat(sum(values),4,1);
 a=values(1,:)+values(2,:);
 b=values(3,:)+values(4,:);
 c=[a;b];
 testLabel=[ones(1, 133),zeros(1,147)];
 plotroc(testLabel,c);

