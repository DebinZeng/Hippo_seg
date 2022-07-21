% ensembling several models to obtain better testing performance                                                                                                                                                                                                                                                                                                                                                                                     % Script to test the model
% --------------------------------------------------------
% Copyright (c) 2018, Debin Zeng
% Licensed under The MIT License
% --------------------------------------------------------

close all; clear all; clc;

% add matcaffe path
addpath(genpath('/home/omnisky/software/3D-Caffe/matlab/')); 
% parameters
use_isotropic = 0;
patchSize = 32;
ita = 4; % control the overlap between different patches
level = 1;
tr_mean = 0;

% set model path 
addpath('./util');
model_folder = '../model_ensemble/';


Data_ID1=textread('../Data_5fold/brain1_ID.txt','%s');
    
Data_ID2=textread('../Data_5fold/brain2_ID.txt','%s');

Data_ID3=textread('../Data_5fold/brain3_ID.txt','%s');

Data_ID4=textread('../Data_5fold/brain4_ID.txt','%s');

Data_ID5=textread('../Data_5fold/brain5_ID.txt','%s');
    
%% all the training data ID
Data_ID=[Data_ID1,Data_ID2,Data_ID3,Data_ID4,Data_ID5];

for foldnum=2:5
snapshot = [model_folder,'snapshot' num2str(foldnum) '/']
result_folder = [snapshot,'ensemble_result/'];
dice_folder=[result_folder,'dice/'];
test_output_folder=[result_folder,'test_output/'];
train_output_folder=[result_folder,'train_output/'];
if ~exist(result_folder,'dir')
     mkdir(result_folder);
end

if ~exist(dice_folder,'dir')
     mkdir(dice_folder);
end

if ~exist(test_output_folder,'dir')
     mkdir(test_output_folder);
end

if ~exist(train_output_folder,'dir')
     mkdir(train_output_folder);
end

% solver= textread([model_folder, 'solver_diceloss' num2str(foldnum) '.prototxt'],'%s');

if foldnum == 1
    fold_test=5:5;
    fold_train=[1,2,3,4];    
elseif  foldnum == 2
    fold_test=4:4;
    fold_train=[1,2,3,5];
elseif foldnum == 3
    fold_test=3:3;
    fold_train=[1,2,4,5];
elseif foldnum == 4
    fold_test=2:2;
    fold_train=[1,3,4,5];
elseif foldnum == 5
    fold_test=1:1;
    fold_train=[2,3,4,5];
end


label_test_avg=cell(3,54);
label_test_vote=cell(3,54);
label_train_avg=cell(3,5,54);
label_train_vote=cell(3,5,54);


for i=1:3
caffe.reset_all
caffe.set_mode_gpu()
caffe.set_device(3)

if i==1
   weights = ['../Vox-ResNet_43_1diceloss_cs32/snapshot' num2str(foldnum) '_diceloss/ADNI_iter_216000.caffemodel']   %%%% 
   model   = ['../Vox-ResNet_43_1diceloss_cs32/deploy_43.prototxt'];
   net = caffe.Net(model, weights, 'test');
elseif i==2
   weights = ['../3D_DenseNet_5/snapshot_' num2str(foldnum) '/ADNI_iter_129600.caffemodel']   %%%% 
   model   = ['../3D_DenseNet_5/deploy_densenet.prototxt'];
   net = caffe.Net(model, weights, 'test'); 
else
   weights = ['../3D_DenseNet_BC/snapshot_' num2str(foldnum) '/ADNI_iter_172800.caffemodel']   %%%% 
   model   = ['../3D_DenseNet_BC/deploy_densenet.prototxt'];
   net = caffe.Net(model, weights, 'test');
end
% dice_test_avg=zeros(4,20);
% dice_test_vote=zeros(4,20);
% dice_train_avg=zeros(4,70);
% dice_train_vote=zeros(4,70);

for fold = fold_test
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
    
    label_test_avg(i,2*(id-1)+j)={avg_label};
    label_test_vote(i,2*(id-1)+j)={vote_label}; 
    
%     %save the test output label_image(avg)
%     avg_label_nii=make_nii(avg_label);
%     save_nii(avg_label_nii, [dice_folder_model,'ADNI_' char(group(j)) num2str(id),'_output_mask_avg.nii'])

    %save the test output label_image(vote)
%     vote_label_nii=make_nii(vote_label);
%     save_nii(vote_label_nii, [dice_folder_model,'ADNI_' char(group(j)) num2str(id),'_output_mask_vote.nii'])

    
    %% compute DICE coefficient
%     groundtruth=load_nii([root_folder, 'ADNI_' char(group(j)) num2str(id) '_label.nii']);
%     dice_test_vote(j,id+1)=2*sum(sum(dot(vote_label,single(groundtruth.img),1)))/(sum(sum(sum(vote_label)))+sum(sum(sum(groundtruth.img))));
%     dice_test_avg(j,id+1)=2*sum(sum(dot(avg_label,single(groundtruth.img),1)))/(sum(sum(sum(avg_label)))+sum(sum(sum(groundtruth.img))));
    
    end
    end
end

for fold = fold_train
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
    
    label_train_avg(i,fold,2*(id-1)+j)={avg_label};
    label_train_vote(i,fold,2*(id-1)+j)={vote_label};   

%     %% compute DICE coefficient
%     groundtruth=load_nii([root_folder_train, 'ADNI_' char(group(j)) num2str(id) '_label.nii']);
%     dice_train_vote(j,id+1)=2*sum(sum(dot(vote_label,single(groundtruth.img),1)))/(sum(sum(sum(vote_label)))+sum(sum(sum(groundtruth.img))));
%     dice_train_avg(j,id+1)=2*sum(sum(dot(avg_label,single(groundtruth.img),1)))/(sum(sum(sum(avg_label)))+sum(sum(sum(groundtruth.img))));
    
    end
    end
end
end
% 
% save([dice_folder,'mean_dice_test_avg.mat'],'mean_dice_test_avg')
% save([dice_folder,'mean_dice_train_avg.mat'],'mean_dice_train_avg')
% save([dice_folder,'mean_dice_test_vote.mat'],'mean_dice_test_vote')
% save([dice_folder,'mean_dice_train_vote.mat'],'mean_dice_train_vote')

%save('filename.mat',B)

dice_test_avg=zeros(1,54);
dice_test_vote=zeros(1,54);
dice_train_avg=zeros(5,54);
dice_train_vote=zeros(5,54);

%%testing
for fold = fold_test
    for id = 1:27
    seg_path_l = [ '../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_L_label_shear.nii'];
    seg_path_r = ['../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_R_label_shear.nii'];
    
    for j=1:2
    vote_label=(label_test_vote{1,2*(id-1)+j}+label_test_vote{2,2*(id-1)+j}+label_test_vote{3,2*(id-1)+j})/3;
    avg_label=(label_test_avg{1,2*(id-1)+j}+label_test_avg{2,2*(id-1)+j}+label_test_avg{3,2*(id-1)+j})/3;
    vote_label(vote_label>=0.5)=1;vote_label(vote_label<0.5)=0;
    avg_label(avg_label>=0.5)=1;avg_label(avg_label<0.5)=0;
%         %%save the test output label_image(avg)
%         avg_label_nii=make_nii(avg_label);
%         save_nii(avg_label_nii, [result_folder,'train_output/ADNI_' char(group(j)) num2str(id),'_output_mask_avg.nii'])

    %%save the test output label_image(vote)
    vote_label_nii=make_nii(vote_label);
    
    if j==1
        save_nii(vote_label_nii, [test_output_folder '/ADNI_' char(fold) Data_ID{id,fold},'_l_output_mask_vote.nii'])
        groundtruth=load_nii(seg_path_l);
    else
        save_nii(vote_label_nii, [test_output_folder '/ADNI_' char(fold) Data_ID{id,fold},'_r_output_mask_vote.nii'])
        groundtruth=load_nii(seg_path_r);
    end
    
    dice_test_vote(2*(id-1)+j)=2*sum(sum(dot(vote_label,single(groundtruth.img),1)))/(sum(sum(sum(vote_label)))+sum(sum(sum(groundtruth.img))));
    dice_test_avg(2*(id-1)+j)=2*sum(sum(dot(avg_label,single(groundtruth.img),1)))/(sum(sum(sum(avg_label)))+sum(sum(sum(groundtruth.img))));
    end
    end
end

for fold = fold_train
    for id = 1:27
        seg_path_l = [ '../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_L_label_shear.nii'];
        seg_path_r = ['../Data_5fold/brain', num2str(fold), '/ADNI_', Data_ID{id,fold}, '_R_label_shear.nii'];
        for j=1:2
        vote_label=(label_train_vote{1,fold,2*(id-1)+j}+label_train_vote{2,fold,2*(id-1)+j}+label_train_vote{3,fold,2*(id-1)+j})/3;
        avg_label=(label_train_avg{1,fold,2*(id-1)+j}+label_train_avg{2,fold,2*(id-1)+j}+label_train_avg{3,fold,2*(id-1)+j})/3;
        vote_label(vote_label>=0.5)=1;vote_label(vote_label<0.5)=0;
        avg_label(avg_label>=0.5)=1;avg_label(avg_label<0.5)=0;
%         %%save the test output label_image(avg)
%         avg_label_nii=make_nii(avg_label);

        %%save the test output label_image(vote)
        vote_label_nii=make_nii(vote_label);
        
        if j==1
            save_nii(vote_label_nii, [train_output_folder '/ADNI_' char(fold)  Data_ID{id,fold},'_l_output_mask_vote.nii'])
            groundtruth=load_nii(seg_path_l);
        else
            save_nii(vote_label_nii, [train_output_folder '/ADNI_' char(fold) Data_ID{id,fold},'_r_output_mask_vote.nii'])
            groundtruth=load_nii(seg_path_r);
        end
        dice_train_vote(fold,2*(id-1)+j)=2*sum(sum(dot(vote_label,single(groundtruth.img),1)))/(sum(sum(sum(vote_label)))+sum(sum(sum(groundtruth.img))));
        dice_train_avg(fold,2*(id-1)+j)=2*sum(sum(dot(avg_label,single(groundtruth.img),1)))/(sum(sum(sum(avg_label)))+sum(sum(sum(groundtruth.img))));
        end
    end
end


csvwrite([dice_folder,'dice_test_avg.csv'], dice_test_avg)
csvwrite([dice_folder,'dice_train_avg.csv'], dice_train_avg)
csvwrite([dice_folder,'dice_test_vote.csv'], dice_test_vote)
csvwrite([dice_folder,'dice_train_vote.csv'], dice_train_vote)

mean_dice_test_avg=sum(dice_test_avg)/54;
mean_dice_test_vote=sum(dice_test_vote)/54;
mean_dice_train_avg=sum(sum(dice_train_avg))/216;
mean_dice_train_vote=sum(sum(dice_train_vote))/216;

fprintf('mean_dice_test_avg #%f, mean_dice_test_vote #%f\n mean_dice_train_avg #%f, mean_dice_train_vote #%f\n',mean_dice_test_avg,mean_dice_test_vote,mean_dice_train_avg,mean_dice_train_vote)

fid=fopen([result_folder, 'dice result.txt'],'a+');
fprintf(fid,'%s\n', ['mean_dice_test_avg ' num2str(mean_dice_test_avg) ', mean_dice_test_vote ' num2str(mean_dice_test_vote) ', mean_dice_train_avg ' num2str(mean_dice_train_avg) ', mean_dice_train_vote ' num2str(mean_dice_train_vote)]);
fclose(fid);

end
% csvwrite([dice_folder,'mean_dice_test_avg.csv'],mean_dice_test_avg)
% csvwrite([dice_folder,'mean_dice_train_avg.csv'],mean_dice_train_avg)
% csvwrite([dice_folder,'mean_dice_test_vote.csv'],mean_dice_test_vote)
% csvwrite([dice_folder,'mean_dice_train_vote.csv'],mean_dice_train_vote)
% 
% 
% fprintf([snapshot,' base_lr:', solver{4}, ' weight_decay:', solver{23}, '\n'])
% [m1,index1]=max(mean_dice_test_avg);
% [m2,index2]=max(mean_dice_test_vote);
% m_test=max(m1,m2);
% if m_test==m1
%     index_test=index1;
%     fprintf('max test dice = %f iter=%d -avg\n',m_test,index_test)
%     t1='avg';
% else
%     index_test=index2;
%     fprintf('max test dice = %f iter=%d -vote\n',m_test,index_test);
%     t1='vote';
% end
% 
% [m3,index3]=max(mean_dice_train_avg);
% [m4,index4]=max(mean_dice_train_vote);
% m_train=max(m3,m4);
% if m_train==m3
%     index_train=index3;
%     fprintf('max train dice = %f iter=%d -avg\n',m_train,index_train)
%     t2='avg';
% else
%     index_train=index4;
%     fprintf('max train dice = %f iter=%d -vote\n',m_train,index_train);
%     t2='vote';
% end


% fid=fopen([model_folder, 'dice result.txt'],'a+');
% fprintf(fid,'%s\n', [solver{4}, ' ', solver{24}, ' ', num2str(m_test), ' ', num2str(index_test), ' ', t1, ' ', num2str(m_train), ' ', num2str(index_train), ' ', t2, ' ', snapshot]);
% fclose(fid);


% iter= 3150:3150:126000;
% figure
% plot(iter,mean_dice_train_avg,'k-','Linewidth',2)
% hold on
% plot(iter,mean_dice_test_avg,'r-','Linewidth',2)
% set(gcf,'PaperUnits','normalized','PaperPosition',[0 0 2 1])
% title([snapshot, 'avg, lr policy:"poly", max iter-', solver{20}],'FontSize',18)
% xlabel('iteration','FontSize',18)
% ylabel('dice','FontSize',18)
% legend('mean dice of training data','mean dice of testing data','Location','east')
% set(gca,'FontSize',16)
% grid on
% saveas(gcf, [result_folder 'avg.jpg'], 'jpg')
% saveas(gcf, [result_folder 'avg.fig'], 'fig')
% 
% figure
% plot(iter,mean_dice_train_vote,'k-','Linewidth',2)
% hold on
% plot(iter,mean_dice_test_vote,'r-','Linewidth',2)
% set(gcf,'PaperUnits','normalized','PaperPosition',[0 0 2 1])
% title([snapshot, 'vote, lr policy:"poly", max iter-',solver{20}],'FontSize',18)
% xlabel('iteration','FontSize',18)
% ylabel('dice','FontSize',18)
% legend('mean dice of training data','mean dice of testing data','Location','east')
% set(gca,'FontSize',16)
% grid on
% saveas(gcf, [result_folder 'vote.jpg'], 'jpg')
% saveas(gcf, [result_folder 'vote.fig'], 'fig')

caffe.reset_all;
