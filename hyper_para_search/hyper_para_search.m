% Optimization of hyperparameters using random search method                                                                                                                                                                                                                                                                                                                                                                                     % Script to test the model
% --------------------------------------------------------
% Copyright (c) 2018, Debin Zeng
% Licensed under The MIT License
% --------------------------------------------------------

% close all; clear all; clc;
% add matcaffe path
addpath(genpath('/home/omnisky/software/3D-Caffe/matlab/'));
addpath('./util');
% 
% generate hyperparameter
% pow1=randi([-35,-15],[20,1]);
% pow2=randi([-50,-30],[20,1]);
% base_lr=10.^(pow1/10);
% L2=10.^(pow2/10);
% csvwrite('./snapshot/hyper_parameter.csv',[base_lr L2]);
hyper_param=csvread('./snapshot/hyper_parameter.csv');
base_lr=hyper_param(:,1);
L2=hyper_param(:,2);
% 
% %initial a matrix to store mean dice of every pair of hyper parameter 
mean_dice_test_avg=zeros(4,1);   %%%%
mean_dice_test_vote=zeros(4,1);
mean_dice_train_avg=zeros(4,1);  %%%%
mean_dice_train_vote=zeros(4,1);
m_test=zeros(20,1);

group = {'AD';'LMCI';'MCI';'Normal'}; 

%set model and data path
model   =  './deploy_densenet.prototxt';
root_folder = '../testing_data/';
root_folder_train ='../training_data/';

% test parameters
use_isotropic = 0;
patchSize = 32;  %crop size
ita = 4; % control the overlap between different patches
level = 1;
tr_mean = 0;

fid1=fopen('./snapshot/max_mean_test_dice.txt','w');

for count=1:20

model_folder=['./snapshot/' 'count_' num2str(count) '/'];    
if ~exist(model_folder,'dir')
     mkdir(model_folder);
end
    
%change the hyper parameter(base_lr and L2)
solver_txt= textread('./solver.prototxt','%s');
solver_txt{4}=num2str(base_lr(count),'%1.7f');
solver_txt{24}=num2str(L2(count),'%1.7f');
solver_txt{28}=['"' model_folder 'ADNI"'];

fid=fopen('./solver.prototxt','w+');
for i=1:32
fprintf(fid,'%s\n', solver_txt{i});
end
fclose(fid);

if count>=2
%% train net
caffe.reset_all
caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.Solver('./solver.prototxt');

solver.step(9600);
% solver.restore('./snapshot')
% iter=solver.iter;

%save the trained model
% train_net = solver.net;
% train_net.save(['./snapshot/count_' num2str(count) '/ADNI_iter_' num2str(iter) '.caffemodel']);
% end
end

for step=1:4
caffe.reset_all
caffe.set_mode_gpu()
caffe.set_device(1)  
    
%% test model
net = caffe.Net(model,[model_folder 'ADNI_iter_' num2str(step*2400) '.caffemodel'], 'test');

%mkdir to save dice coef
dice_folder=[model_folder 'dice_' num2str(step*2400) '/'];
if ~exist(dice_folder,'dir')
     mkdir(dice_folder);
end

%initial a matrix to store dice coef of every person 
dice_test_avg=zeros(4,4);
dice_test_vote=zeros(4,4);
dice_train_avg=zeros(4,16);
dice_train_vote=zeros(4,16);

for j =1:4   
fprintf('test model_hyper_para_count #%d, test group #%s\n',count,char(group(j)))

%%testing    
group_list_path = ['../data_info/', char(group(j)), '_test.txt'];
group_list= textread(group_list_path,'%s');
num=length(group_list);

for id = 0:(2*num-1)
    %% read data and preprocessing
    tic;
    fprintf('test sample #%d\n',id)
    pred_list = [];
    vol_path = [root_folder, 'ADNI_' char(group(j)) num2str(id) '.nii'];
    vol_src = load_nii(vol_path);
    vol_data = pre_process_isotropic(vol_src,[],use_isotropic,id);  %only normalize the intensity here
    data = vol_data - tr_mean;
    
    %% average fusion scheme
    score_map = generate_score_map(net, data, patchSize, ita,level);
    [~,res_label] = max(score_map,[],4);
    avg_label = res_label -1;
    %avg_label = GetLabelfromScore(score_map);

    %% major voting scheme
    [patch_list, r, c, h] = partition_Img(data, patchSize, ita);
	   
    for k = 1 : length(patch_list)
        crop_data = patch_list{k};
        res = net.forward({crop_data});
        res_L2 = res{level};
        %res_label = GetLabelfromScore(res_L2);
        [~,res_label] = max(res_L2,[],4);
        pred_list{k} = res_label-1;
    end
    [fusion_Img_L2, vote_label] = patches2Img_vote(pred_list, r, c, h, patchSize, ita); 
    
    %% post-processing
    if use_isotropic == 1
        vote_label = imresize3d(vote_label,[],vol_src.hdr.dime.dim(2:4),'nearest','bound');
        avg_label = imresize3d(avg_label,[],vol_src.hdr.dime.dim(2:4),'nearest','bound');
    end
    
    %% remove minor connected components
    vote_label = RemoveMinorCC(vote_label,0.2);
    avg_label = RemoveMinorCC(avg_label,0.2);
    toc; 
    
%     %save the test output label_image(avg)
%     avg_label_nii=make_nii(avg_label);
%     save_nii(avg_label_nii, [dice_folder_model,'ADNI_' char(group(j)) num2str(id),'_output_mask_avg.nii'])

    %save the test output label_image(vote)
%     vote_label_nii=make_nii(vote_label);
%     save_nii(vote_label_nii, [dice_folder_model,'ADNI_' char(group(j)) num2str(id),'_output_mask_vote.nii'])

    
    %% compute DICE coefficient
    groundtruth=load_nii([root_folder, 'ADNI_' char(group(j)) num2str(id) '_label.nii']);
    dice_test_vote(j,id+1)=2*sum(sum(dot(vote_label,single(groundtruth.img),1)))/(sum(sum(sum(vote_label)))+sum(sum(sum(groundtruth.img))));
    dice_test_avg(j,id+1)=2*sum(sum(dot(avg_label,single(groundtruth.img),1)))/(sum(sum(sum(avg_label)))+sum(sum(sum(groundtruth.img))));
    
end

%%training
group_list_path_train = ['../data_info/', char(group(j)), '_train.txt'];
group_list_train= textread(group_list_path_train,'%s');
num_train=length(group_list_train);

for id = 0:(2*num_train-1)
    %% read data and preprocessing
    tic;
    fprintf('train sample #%d\n',id)
    pred_list = [];
    vol_path = [root_folder_train, 'ADNI_' char(group(j)) num2str(id) '.nii'];
    vol_src = load_nii(vol_path);
    vol_data = pre_process_isotropic(vol_src,[],use_isotropic,id);  %only normalize the intensity here
    data = vol_data - tr_mean;
 
    %% average fusion scheme
    score_map = generate_score_map(net, data, patchSize, ita,level);
    [~,res_label] = max(score_map,[],4);
    avg_label = res_label -1;
    %avg_label = GetLabelfromScore(score_map);

    %% major voting scheme
    [patch_list, r, c, h] = partition_Img(data, patchSize, ita);
	
    for k = 1 : length(patch_list)
        crop_data = patch_list{k};
        res = net.forward({crop_data});
        res_L2 = res{level};
        %res_label = GetLabelfromScore(res_L2);
        [~,res_label] = max(res_L2,[],4);
        pred_list{k} = res_label-1;
    end
    [fusion_Img_L2, vote_label] = patches2Img_vote(pred_list, r, c, h, patchSize, ita); 
    
    %% post-processing
    if use_isotropic == 1
        vote_label = imresize3d(vote_label,[],vol_src.hdr.dime.dim(2:4),'nearest','bound');
        avg_label = imresize3d(avg_label,[],vol_src.hdr.dime.dim(2:4),'nearest','bound');
    end
    
    %% remove minor connected components
    vote_label = RemoveMinorCC(vote_label,0.2);
    avg_label = RemoveMinorCC(avg_label,0.2);
    toc;   

    %% compute DICE coefficient
    groundtruth=load_nii([root_folder_train, 'ADNI_' char(group(j)) num2str(id) '_label.nii']);
    dice_train_vote(j,id+1)=2*sum(sum(dot(vote_label,single(groundtruth.img),1)))/(sum(sum(sum(vote_label)))+sum(sum(sum(groundtruth.img))));
    dice_train_avg(j,id+1)=2*sum(sum(dot(avg_label,single(groundtruth.img),1)))/(sum(sum(sum(avg_label)))+sum(sum(sum(groundtruth.img))));
    
end
end

csvwrite([dice_folder,'dice_test_avg.csv'], dice_test_avg)
csvwrite([dice_folder,'dice_train_avg.csv'], dice_train_avg)
csvwrite([dice_folder,'dice_test_vote.csv'], dice_test_vote)
csvwrite([dice_folder,'dice_train_vote.csv'], dice_train_vote)
 
%compute mean dice of all people
mean_dice_test_avg(step)=sum(sum(dice_test_avg))/12;
mean_dice_test_vote(step)=sum(sum(dice_test_vote))/12;
mean_dice_train_avg(step)=sum(sum(dice_train_avg))/48;
mean_dice_train_vote(step)=sum(sum(dice_train_vote))/48;

%fprintf('count_num #%d base_lr:%f weight_decay:%f\n mean_dice_test_avg #%f, mean_dice_test_vote #%f\n mean_dice_train_avg #%f, mean_dice_train_vote #%f\n', ...
%    count,solver_txt{4},solver_txt{24},mean_dice_test_avg(count),mean_dice_test_vote(count),mean_dice_train_avg(count),mean_dice_train_vote(count))

end
csvwrite(['./snapshot/count_' num2str(count) '/mean_dice_test_avg.csv'],mean_dice_test_avg)
csvwrite(['./snapshot/count_' num2str(count) '/mean_dice_test_vote.csv'],mean_dice_test_vote)
csvwrite(['./snapshot/count_' num2str(count) '/mean_dice_train_avg.csv'],mean_dice_train_avg)
csvwrite(['./snapshot/count_' num2str(count) '/mean_dice_train_vote.csv'],mean_dice_train_vote)

m1=max(mean_dice_test_avg);
m2=max(mean_dice_test_vote);
m_test(count)=max(m1,m2);


fprintf(fid1,'count_num #%d base_lr:%s weight_decay:%s\n mean_dice_test_avg #%s\n', count,solver_txt{4},solver_txt{24});
fprintf(fid1,'max_mean_test_dice#%f\n',m_test(count));

end
fclose(fid1);

[max_test,index]=max(m_test);


caffe.reset_all;
