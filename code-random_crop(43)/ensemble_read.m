model_folder = '../ensemble-43-diceloss/';

solver= textread([model_folder, 'solver.prototxt'],'%s');snapshot = [model_folder,'snapshot/count',num2str(count),'/'];

count=12;
result_folder = [snapshot,'result/'];
dice_folder=[result_folder,'dice/'];

mean_dice_test_avg=csvread([dice_folder,'mean_dice_test_avg.csv']);
mean_dice_train_avg=csvread([dice_folder,'mean_dice_train_avg.csv']);
mean_dice_test_vote=csvread([dice_folder,'mean_dice_test_vote.csv']);
mean_dice_train_vote=csvread([dice_folder,'mean_dice_train_vote.csv']);


fprintf([snapshot,' base_lr:', solver{4}, ' weight_decay:', solver{23}, '\n'])
[m1,index1]=max(mean_dice_test_avg);
[m2,index2]=max(mean_dice_test_vote);
m_test=max(m1,m2);
if m_test==m1
    index_test=index1;
    fprintf('max test dice = %f iter=%d -avg\n',m_test,index_test)
    m_train=mean_dice_train_avg(index_test);
    fprintf(' train dice = %f iter=%d -avg\n',m_train,index_test)
    t1='avg';
else
    index_test=index2;
    fprintf('max test dice = %f iter=%d -vote\n',m_test,index_test);
    m_train=mean_dice_train_vote(index_test);
    fprintf(' train dice = %f iter=%d -vote\n',m_train,index_test)
    t1='vote';
end

fid=fopen([model_folder, 'dice result.txt'],'a+');
fprintf(fid,'%s\n', [solver{4}, ' ', solver{23}, ' ', num2str(m_test), ' ', num2str(index_test), ' ', t1, ' ', num2str(m_train), ' ', num2str(index_test), ' ', t1, ' ', snapshot]);
fclose(fid);


iter= 3150:3150:37800;
figure
plot(iter,mean_dice_train_avg,'k-','Linewidth',2)
hold on
plot(iter,mean_dice_test_avg,'r-','Linewidth',2)
set(gcf,'PaperUnits','normalized','PaperPosition',[0 0 2 1])
title([snapshot, 'avg, lr policy:"poly", max iter-126000'],'FontSize',18)
xlabel('iteration','FontSize',18)
ylabel('dice','FontSize',18)
legend('mean dice of training data','mean dice of testing data','Location','east')
set(gca,'FontSize',16)
grid on
saveas(gcf, [result_folder 'avg.jpg'], 'jpg')
saveas(gcf, [result_folder 'avg.fig'], 'fig')

figure
plot(iter,mean_dice_train_vote,'k-','Linewidth',2)
hold on
plot(iter,mean_dice_test_vote,'r-','Linewidth',2)
set(gcf,'PaperUnits','normalized','PaperPosition',[0 0 2 1])
title([snapshot, 'vote, lr policy:"poly", max iter-126000'],'FontSize',18)
xlabel('iteration','FontSize',18)
ylabel('dice','FontSize',18)
legend('mean dice of training data','mean dice of testing data','Location','east')
set(gca,'FontSize',16)
grid on
saveas(gcf, [result_folder 'vote.jpg'], 'jpg')
saveas(gcf, [result_folder 'vote.fig'], 'fig')
