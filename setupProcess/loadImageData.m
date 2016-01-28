function dataset=loadImageData(path, width, height)
if ~exist('dcm', 'dir')
    mkdir('dcm');
end
if ~exist('dcm/dataset.mat', 'file')
    if length(path)==0
        error('no path...');
    end
    dataset=cell(length(path),1);
    for index=1:length(path)
        detailPath=path{index};
        files=dir(fullfile(detailPath, '*.jpg'));
        dataset{index}.img=zeros(width*height, length(files));
        for i=1:length(files)
            img=imread([detailPath, '/', files(i).name]);
            img=double(img);
            img=img/max(max(img));
            dataset{index}.img(:, i)=img(:);
        end
    end
    save('dcm/dataset.mat', 'dataset', '-v7.3');
    disp('dataset have saved.');
else
    disp('load data form dcm/');
    load dcm/dataset.mat;
end
end