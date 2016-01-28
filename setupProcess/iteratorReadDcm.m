function matdat = iteratorReadDcm(superPath)
subDir =  dir(fullfile(superPath,'*'));
matdat=cell(1,length(subDir)-2);
j=1;
for i=1:length(subDir)
    fileName=subDir(i).name;
    if strcmp(fileName,'.') || strcmp(fileName, '..')
        continue;
    end
    fprintf('%s:\tprocess %s ... \n', mfilename, fileName);
    matdat{j}=dcm2mat([superPath, '/', fileName]);
    j = j+1;
end
end
