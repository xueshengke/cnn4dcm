%% 数据预处理，将dcm转为mat
function I=dcm2mat(filePath)
eachDir =  dir(fullfile(filePath, '*.dcm'));
dcmNum=length(eachDir);
records = zeros(dcmNum);
for j=1:dcmNum
    dcmFileName = eachDir(j).name;
    splitStrArr = regexp(dcmFileName, '\.', 'split');
    records(j)=int32(str2double(splitStrArr{end-1}));
end
[~,sortedIndex] = sort(records);
for k= 1:length(eachDir)
    sortedDir(k) = eachDir(sortedIndex(k));
end
I = zeros(dcmNum, 512, 512);
for k=1:length(sortedDir)
    fileAbsolutePath = [filePath,'/', dcmFileName];
    info = dicominfo(fileAbsolutePath);
    I(k,:,:)=dicomread(info);
end
end