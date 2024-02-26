load('covtypecondnum=1000000000.mat');
thre=80;

figure

semilogy(1:N,error_gd(1:N),'r');
hold on
semilogy(1:length(error_LM_AA1),error_LM_AA1,'g');
hold on
semilogy(1:length(error_rna5),error_rna5,'b');
hold on
semilogy(1:length(error_rna10),error_rna10,'c');
hold on
semilogy(1:length(error_rna20),error_rna20,'y');
%legend({'Gradient Descend', 'LM-AA','RNA k=5','RNA k=10','RNA k=20'},'location','SW')
xlabel('Iteration');
ylabel('(F-F*)/F*');
saveas(gcf,strcat(dataset_name,'_',num2str(condnum),'_iteration.pdf'));

figure
for i=1:length(time_gd)
    if time_gd(i)>thre
        index=i;
        break;
    end
end
semilogy(time_gd(1:index),error_gd(1:index),'r');
hold on
for i=1:length(time_LM_AA1)
    if time_LM_AA1(i)>thre
        index=i;
        break;
    end
end
semilogy(time_LM_AA1(1:index),error_LM_AA1(1:index),'g');
hold on
for i=1:length(time_rna5)
    if time_rna5(i)>thre
        index=i;
        break;
    end
end
semilogy(time_rna5(1:index),error_rna5(1:index),'b');
hold on
for i=1:length(time_rna10)
    if time_rna10(i)>thre
        index=i;
        break;
    end
end
semilogy(time_rna10(1:index),error_rna10(1:index),'c');
hold on
for i=1:length(time_rna20)
    if time_rna20(i)>thre
        index=i;
        break;
    end
end
semilogy(time_rna20(1:index),error_rna20(1:index),'y');
hold on
%legend({'Gradient Descend', 'LM-AA','RNA k=5','RNA k=10','RNA k=20'},'location','SW')
xlabel('time(s)');
ylabel('(F-F*)/F*');
axis([0,thre,1e-12,1e0]);
saveas(gcf,strcat(dataset_name,'_',num2str(condnum),'_times.pdf'))
%  figure
% % 
% semilogy(time_gd,error_gd_g,'r');
% hold on
% semilogy(time_LM_AA0,error_LM_AA_g0,'b');
% hold on
% semilogy(time_rna,error_rna_g,'g');
% hold on
% semilogy(time_LM_AA1,error_LM_AA_g1,'c');
% hold on
% semilogy(time_AAE,error_AAE_g,'m');
% legend({'Gradient Descend', 'LM-AA 0','RNA','LM-AA 1','AA-Energy'},'location','NE')
% xlabel('time(s)');
% ylabel('(AA Residual');
 