function getInputData(path)
files =  dir(fullfile(path), '*.mat');
for i=1:length(files)
    fileName=file(i).name;
    fprintf('%s:%s\n', mfilename, fileName);
    load([path, '/', fileName]);
    fprintf('%s\n');
    size(I);
    allData(i,:,:,:)=I(:,:,:);
end
save allData.mat allData
end
