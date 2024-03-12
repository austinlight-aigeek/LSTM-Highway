% time-series forecasting with rnn
clc
clear all
close all

load('shanghai_gold_price.mat');
% load('monthly_gold_price.mat');

E = mean(data);
mu = std(data);
data = (data - E)/mu;

m = 1;      % input dimension
l = 128;    % hidden node dimension
n = 1;      % target dimension
bSize = 64;     % length of LSTM network

data_size = numel(data);
%train_size = floor(numel(data)*0.95/bSize)*bSize + 1;
train_size = 201;
test_size = numel(data) - train_size;
train_data = data(1:train_size);
test_data = data(train_size+1: end);

Wf = 0.01*randn(l,m);   Rf = 0.01*randn(l,l);   bf = 0.01*randn(l,1);
Wi = 0.01*randn(l,m);   Ri = 0.01*randn(l,l);   bi = 0.01*randn(l,1);
Wg = 0.01*randn(l,m);   Rg = 0.01*randn(l,l);   bg = 0.01*randn(l,1);
Wo = 0.01*randn(l,m);   Ro = 0.01*randn(l,l);   bo = 0.01*randn(l,1);
V = 0.01*randn(n,l);    b = 0.01*randn(n,1);

mWf = zeros(l,m);   mRf = zeros(l,l);   mbf = zeros(l,1);
mWi = zeros(l,m);   mRi = zeros(l,l);   mbi = zeros(l,1);
mWg = zeros(l,m);   mRg = zeros(l,l);   mbg = zeros(l,1);
mWo = zeros(l,m);   mRo = zeros(l,l);   mbo = zeros(l,1);
mV = zeros(n,l);    mb = zeros(n,1);

epoch_num = 150;
learning_rate = 0.001;
mnt_rate = 0.9;

numBatch = floor(train_size/bSize);     % number of possible full mini-batches
bList = 1:bSize:(numBatch)*bSize+1;   % min-batch index list

for epoch = 1:epoch_num
    h0 = zeros(l,1);
    c0 = zeros(l,1);
    L = 0;
    for bStart = bList
        bEnd = min(bStart+bSize-1, train_size-1);
        
        inputs = train_data(bStart: bEnd);
        targets = train_data(bStart+1: bEnd+1);
        
        [dWf,dRf,dbf,dWi,dRi,dbi,dWg,dRg,dbg,dWo,dRo,dbo,dV,db, h0, c0, loss] = ...
            lstm(Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,inputs, targets, h0, c0);

        mWf = mnt_rate*mWf - learning_rate*dWf;
        mWi = mnt_rate*mWi - learning_rate*dWi;
        mWg = mnt_rate*mWg - learning_rate*dWg;
        mWo = mnt_rate*mWo - learning_rate*dWo;
        
        mRf = mnt_rate*mRf - learning_rate*dRf;
        mRi = mnt_rate*mRi - learning_rate*dRi;
        mRg = mnt_rate*mRg - learning_rate*dRg;
        mRo = mnt_rate*mRo - learning_rate*dRo;
        
        mbf = mnt_rate*mbf - learning_rate*dbf;
        mbi = mnt_rate*mbi - learning_rate*dbi;
        mbg = mnt_rate*mbg - learning_rate*dbg;
        mbo = mnt_rate*mbo - learning_rate*dbo;
        
        mV = mnt_rate*mV - learning_rate*dV;
        mb = mnt_rate*mb - learning_rate*db;

        Wf = Wf + mWf;  Rf = Rf + mRf; bf = bf + mbf;
        Wi = Wi + mWi;  Ri = Ri + mRi; bi = bi + mbi;
        Wg = Wg + mWg;  Rg = Rg + mRg; bg = bg + mbg;
        Wo = Wo + mWo;  Ro = Ro + mRo; bo = bo + mbo;
        V = V + mV;     b = b + mb;
        
        L = L + loss;
    end
    
    if (~mod(epoch, 10) || epoch == 1)
        str = sprintf('epoch: %d, loss: %f', epoch, L);
        disp(str);
    end
end

h0 = zeros(l,1);
c0 = zeros(l,1);
[hend, cend, yy] = lstm_forward(...
    Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,train_data, h0, c0);

pred = zeros(n, test_size);
for j = 1:test_size
    [hend, cend, pred(:,j)] = lstm_forward(...
        Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,V,b,test_data(:,j), hend, cend);
end

pred = pred*mu + E;
targets = test_data*mu +E;
xx = 1:test_size;
plot(xx, targets, xx, pred);

RMSE = sqrt(mean((targets - pred).^2))
