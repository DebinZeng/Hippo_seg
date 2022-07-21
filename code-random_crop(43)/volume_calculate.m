% Compute hippocampal volume
% --------------------------------------------------------
% Copyright (c) 2018, Debin Zeng
% Licensed under The MIT License
% --------------------------------------------------------

Data_ID1=textread('../Data_5fold/brain1_ID.txt','%s');
Data_ID2=textread('../Data_5fold/brain2_ID.txt','%s');
Data_ID3=textread('../Data_5fold/brain3_ID.txt','%s');
Data_ID4=textread('../Data_5fold/brain4_ID.txt','%s');
Data_ID5=textread('../Data_5fold/brain5_ID.txt','%s');
    
%% all the training data ID
Data_ID=[Data_ID1,Data_ID2,Data_ID3,Data_ID4,Data_ID5];

volume_human=zeros(5,54);
for fold=1:5

    for id = 1:27
    
    for j=1:2
        if j==1
            direction='L';
        else
            direction='R';
        end
        
        
        seg_path = ['../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_' direction '_label_shear.nii'];

        groundtruth=load_nii(seg_path);
        volume_human(6-fold,2*(id-1)+j)=sum(sum(sum(groundtruth.img)));
        
    end   
    end
end

 result_folder = ['../Data_5fold/volume/'];   
 if ~exist(result_folder,'dir')
     mkdir(result_folder);
 end
 
 csvwrite([result_folder,'volume_human.csv'],volume_human)
