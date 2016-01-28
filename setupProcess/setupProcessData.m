clc;
path='/home/jql/data/origindata'

djj = iteratorReadDcm([path, '/', 'djj']);
size(djj)
save('djj','djj', '-v7.3');
fprintf('save djj.mat\n');

xjj = iteratorReadDcm([path, '/', 'xjj']);
size(xjj)
save('xjj','xjj','-v7.3');
fprintf('save xjj.mat\n');

numDjj=size(djj,2)
numXjj=size(xjj,2)
num=numDjj+numXjj;
matdat=cell(1, num);
for i=1:num
	if i<=numDjj
		matdat{i}=djj{i};
	else
		matdat{i}=xjj{i-numDjj};
	end
end
size(matdat)

save('matdat', 'matdat', '-v7.3');
fprintf('save matdat.mat\n');

formData=formatData(matdat,[numDjj, numXjj], 50);
save('formData', 'formData', '-v7.3');

