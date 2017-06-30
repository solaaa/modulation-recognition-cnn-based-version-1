clear all
source = load('.\data_sample_QPSK_SNRmixed.txt');
data = cell(200,1);




%% 按SNR筛选
%

iter = 1;
SNR = 2;

for j=1:800
     if source(1+129*(j-1),2) == SNR
        data{iter,1} = source(2 + (j-1)*129 : 129*j, 1:2);
        iter = iter + 1;
        disp(iter)
     end
end
% 星座图
x = [];
y = [];
NUM = 5;

figure(1)
x = [x; data{NUM}(1:end, 1)];
y = [y; data{NUM}(1:end, 2)];
plot(x, y);
hold on;
plot(x(1:8:end), y(1:8:end), 'r.');

% IQ序列
figure(2)
order = 0.15;
t = 0.1:0.1:12.8;
subplot(2,1,1); 
plot(t, x);hold on;
x2 = lwlr(t', t', x, order);
plot(t, x2, 'r');


subplot(2,1,2);
plot(t, y);hold on;
y2 = lwlr(t', t', y, order);
plot(t, y2, 'r');


% 拟合后的星座图
figure(4)
plot(x2, y2);
hold on;
plot(x2(1:8:end), y2(1:8:end), 'r.');

% t-f-A 时频图
% figure(3)
% subplot(3,1,1);
% spectrogram(x,16,10,16,0.01);
% subplot(3,1,2);
% spectrogram(y,16,10,16,0.01);
% subplot(3,1,3);
% z = x + y*1i;
% spectrogram(z,16,10,16,0.01);


%%
% 
% 将多个同SNR序列链接
%
% iter = 1;
% SNR = 16;
% 
% for j=1:800
%      if source(1+129*(j-1),2) == SNR
%         data{iter,1} = source(2 + (j-1)*129 : 129*j, 1:2);
%         iter = iter + 1;
%         disp(iter)
%      end
% end
% 
% x = [];
% y = [];
% 
% NUM = 10;
% for i = 1:NUM
%     x = [x; data{i}(1:end, 1)];
%     y = [y; data{i}(1:end, 2)];
% end
% 
% % 星座图
% figure(4)
% plot(x, y);
% hold on;
% plot(x(1:4:end), y(1:4:end), 'r.');
% 
% % IQ序列
% figure(5)
% t = 1:NUM*128;
% subplot(2,1,1); 
% plot(t, x); hold on;
% plot(t, x, 'r.');
% subplot(2,1,2);
% plot(t, y); hold on;
% plot(t, y, 'r.')
% 
% % t-f-A 时频图
% figure(6)
% subplot(3,1,1);
% spectrogram(x,16,10,16,0.01);
% subplot(3,1,2);
% spectrogram(y,16,10,16,0.01);
% subplot(3,1,3);
% z = x + y*1i;
% spectrogram(z,16,10,16,0.01);