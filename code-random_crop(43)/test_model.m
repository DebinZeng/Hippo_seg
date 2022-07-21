% Script to test the model
% --------------------------------------------------------
% Copyright (c) 2018, Debin Zeng
% Licensed under The MIT License
% --------------------------------------------------------
%before run:
%1.modify the model_folder and snapshot folder
%2.modify the iter-num, mean_dice num
%3.modify GPU device
%4.modify figure title
% --------------------------------------------------------

close all; clear all; clc;

% add matcaffe path
addpath(genpath('/Data_HD16/software/3D-Caffe/matlab'));
% parameters
use_isotropic = 0;
%patchSize = 64;
%ita = 4; # control the overlap between different patches
level = 1;
tr_mean = 0;

Data_ID1=textread('../Data_5fold/brain1_ID.txt','%s');
Data_ID2=textread('../Data_5fold/brain2_ID.txt','%s');
Data_ID3=textread('../Data_5fold/brain3_ID.txt','%s');
Data_ID4=textread('../Data_5fold/brain4_ID.txt','%s');
Data_ID5=textread('../Data_5fold/brain5_ID.txt','%s');
    
%% all the training data ID
Data_ID=[Data_ID1,Data_ID2,Data_ID3,Data_ID4,Data_ID5];

% set model path 
addpath('./util');
model_folder = '../3D-DCFCN-BC-131/';
model   = [model_folder, 'deploy_densenet.prototxt'];

model_id = [2160:2160:108000];  %%%%%%%%%%%%
model_num = length(model_id);

gpu_device=3;

for s=[1 3]
snapshot = [model_folder,'snapshot_' num2str(s) '/']     %%%%%%%%%

solver= textread([model_folder, 'solver_densenet_' num2str(s) '.prototxt'],'%s');%%%%%%%%%

%% set save path
result_folder = [snapshot,'result/'];
dice_folder=[result_folder,'dice' num2str(s) '/'];

if ~exist(result_folder,'dir')
     mkdir(result_folder);
end

if ~exist(dice_folder,'dir')
     mkdir(dice_folder);
end

%%
mean_dice_test=zeros(model_num,1);  
mean_dice_train=zeros(model_num,1);  

fold_test = s:s;       %%%%%%%%%%%%%%%%%
fold_train = [1,2,3,4,5];%%%%%%%%%%%%%%%%%
fold_train(s) = []; 

for i= model_id
%i=86400;
dice_folder_model=[dice_folder,'dice_iter_' num2str(i) '/'];

if ~exist(dice_folder_model,'dir')
     mkdir(dice_folder_model);
end
% 
% if s==2 && i<=172800 %load dice coefficient which have been computed
% 
%  dice_test_avg=csvread([result_folder, 'dice/dice_iter_' num2str(i) '/dice_test_avg.csv']);
%  dice_test_vote=csvread([result_folder, 'dice/dice_iter_' num2str(i) '/dice_test_vote.csv']);
%  dice_train_avg=csvread([result_folder, 'dice/dice_iter_' num2str(i) '/dice_train_avg.csv']);
%  dice_train_vote=csvread([result_folder, 'dice/dice_iter_' num2str(i) '/dice_train_vote.csv']);    
% 
% else
caffe.reset_all
caffe.set_mode_gpu()
caffe.set_device(gpu_device)     %%%%%%%%%%%%%%%%%
weights = [snapshot, 'ADNI_iter_' num2str(i) '.caffemodel']   

% if ~exist([result_folder 'avg/'],'dir')
%     mkdir([result_folder 'avg/']);
% end
% if ~exist([result_folder 'voting/'],'dir')
%     mkdir([result_folder 'voting/']);
% end

% prediction
if(~exist(weights,'file'))
    disp('waiting...')
    pause(3240);
end

net = caffe.Net(model, weights, 'test');

dice_test=zeros(1,54);
dice_train=zeros(5,54);
volume_test_vote=zeros(1,54);
%% testing

for fold=fold_test
    for id = 1:27
    
    img_path_l = [ '../Data_5fold/brain', num2str(fold),'/ADNI_', Data_ID{id,fold}, '_L_shear.nii'];
    seg_path_l = [ '../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_L_label_shear.nii'];
    img_path_r = [ '../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_R_shear.nii'];
    seg_path_r = ['../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_R_label_shear.nii'];
     
    %% read data and preprocessing
    fprintf('test sample #%d #%d\n',fold,id)
    for j=1:2
    tic;   
    if j==1
        vol_path = img_path_l;
    else
        vol_path = img_path_r;
    end   
    pred_list = [];
    vol_src = load_nii(vol_path);
    vol_data = pre_process_isotropic(vol_src,[],use_isotropic,id);  %only normalize the intensity here
    data = vol_data - tr_mean;
    
    res = net.forward({single(data)});
    res_score = res{level};
    [~,res_label] = max(res_score,[],4);
    mask = res_label -1;
%     mask=res_score(:,:,:,2);
%     mask(mask>=0.5)=1;
%     mask(mask<0.5)=0;
%    mask_nii=make_nii(mask);
%    save_nii(mask_nii, [result_folder,'ADNI_' group num2str(id),'_output_mask.nii'])
    
    %%compute DICE coefficient
    if j==1
       mask_nii=make_nii(mask);
       %save_nii(mask_nii, [dice_folder_model,'ADNI_', Data_ID{id,fold},'_L_output_mask_vote.nii']); 
       groundtruth=load_nii(seg_path_l);
    else
       mask_nii=make_nii(mask);
       %save_nii(mask_nii, [dice_folder_model,'ADNI_', Data_ID{id,fold},'_R_output_mask_vote.nii']) 
       groundtruth=load_nii(seg_path_r);
    end
    volume_test_vote(2*(id-1)+j)=sum(sum(sum(mask)));
    
    dice_test(2*(id-1)+j)=2*sum(sum(dot(mask,single(groundtruth.img),1)))/(sum(sum(sum(mask)))+sum(sum(sum(groundtruth.img))));
    end
    end
end
csvwrite([dice_folder_model,num2str(s),'_volume_test_vote.csv'],volume_test_vote)
%training
for fold= fold_train
    for id = 1:27
    %% read data and preprocessing
   
    img_path_l = [ '../Data_5fold/brain', num2str(fold),'/ADNI_', Data_ID{id,fold}, '_L_shear.nii'];
    seg_path_l = [ '../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_L_label_shear.nii'];
    img_path_r = [ '../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_R_shear.nii'];
    seg_path_r = ['../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_R_label_shear.nii'];
    
    fprintf('train sample #%d-#%d\n',fold,id)
    for j=1:2
    tic;    
    if j==1
        vol_path = img_path_l;
    else
        vol_path = img_path_r;
    end
    
    pred_list = [];
    vol_src = load_nii(vol_path);
    vol_data = pre_process_isotropic(vol_src,[],use_isotropic,id);  %only normalize the intensity here
    data = vol_data - tr_mean;
    
    res = net.forward({single(data)});
    res_score = res{level};   
    [~,res_label] = max(res_score,[],4);
    mask = res_label -1;    
%     mask=res_score(:,:,:,2);
%     mask(mask>=0.5)=1;
%     mask(mask<0.5)=0;
%    mask_nii=make_nii(mask);
%    save_nii(mask_nii, [result_folder,'ADNI_' group num2str(id),'_output_mask.nii'])
    
    %%compute DICE coefficient
    if j==1
        groundtruth=load_nii(seg_path_l);
    else
        groundtruth=load_nii(seg_path_r);
    end
    dice_train(fold,2*(id-1)+j)=2*sum(sum(dot(mask,single(groundtruth.img),1)))/(sum(sum(sum(mask)))+sum(sum(sum(groundtruth.img))));

    end
    end
end

csvwrite([dice_folder_model,'dice_train.csv'], dice_train)
csvwrite([dice_folder_model,'dice_test.csv'], dice_test)

mean_dice_test(i/model_id(1))=sum(dice_test)/54;
mean_dice_train(i/model_id(1))=sum(sum(dice_train))/216;
fprintf('test model_iter #%d\n mean_dice_test #%f\n mean_dice_train #%f\n', i,mean_dice_test(i/model_id(1)),mean_dice_train(i/model_id(1)))
end

csvwrite([dice_folder,'mean_dice_train.csv'],mean_dice_train)
csvwrite([dice_folder,'mean_dice_test.csv'],mean_dice_test)

iter= model_id;
figure
plot(iter,mean_dice_train,'k-','Linewidth',2)
hold on
plot(iter,mean_dice_test,'r-','Linewidth',2)
set(gcf,'PaperUnits','normalized','PaperPosition',[0 0 2 1])
title([snapshot, 'avg, lr policy:"poly", max iter-' num2str(model_id(end))],'FontSize',18)
xlabel('iteration','FontSize',18)
ylabel('dice','FontSize',18)
legend('mean dice of training data','mean dice of testing data','Location','east')
set(gca,'FontSize',16)
grid on
saveas(gcf, [result_folder 'avg.jpg'], 'jpg')
saveas(gcf, [result_folder 'avg.fig'], 'fig')
end
%saveas(gcf,'3D-DSN base lr-1e-2.5.jpg')
caffe.reset_all;
