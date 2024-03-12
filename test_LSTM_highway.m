% time-series forecasting with rnn
clc
clear all
close all

% load('shanghai_gold_price.mat');
load('monthly_gold_price.mat');
% data = chickenpox_dataset;
% data = [data{:}];
% x = (-50:0.2:50);
% data = sinc(2*x)-0.4*cos(3*x)+0.2*sin(2*x-50)-0.3*sin(0.5*x-5);

E = mean(data);
mu = std(data);

m = 1;      % input dimension
l = 128;    % hidden node dimension
n = 1;      % target dimension
bSize = 64;     % length of LSTM network

data_size = numel(data);
% train_size = floor(numel(data)*0.9/bSize)*bSize + 1;
train_size = 200;

XTrain = data(1:train_size);
YTrain = data(2:train_size+1);
XTest = data(train_size+1:end-1);
YTest = data(train_size+2:end);

test_size = numel(XTest);

XTrain = (XTrain - E) / mu;
YTrain = (YTrain - E) / mu;
XTest = (XTest - E) / mu;

% ======================== weights initialization =========================
Wf = 0.01*randn(l,m);   Rf = 0.01*randn(l,l);   bf = 0.01*randn(l,1);
Wi = 0.01*randn(l,m);   Ri = 0.01*randn(l,l);   bi = 0.01*randn(l,1);
Wg = 0.01*randn(l,m);   Rg = 0.01*randn(l,l);   bg = 0.01*randn(l,1);
Wo = 0.01*randn(l,m);   Ro = 0.01*randn(l,l);   bo = 0.01*randn(l,1);

Wp = 0.01*randn(l,l);   bp = 0.01*randn(l,1);
Wr = 0.01*randn(l,l);   br = 0.01*randn(l,1);
Wq = 0.01*randn(l,l);   bq = 0.01*randn(l,1);

V = 0.01*randn(n,l);    b = 0.01*randn(n,1);

mWf = zeros(l,m);   mRf = zeros(l,l);   mbf = zeros(l,1);
mWi = zeros(l,m);   mRi = zeros(l,l);   mbi = zeros(l,1);
mWg = zeros(l,m);   mRg = zeros(l,l);   mbg = zeros(l,1);
mWo = zeros(l,m);   mRo = zeros(l,l);   mbo = zeros(l,1);
mWp = zeros(l,m);   mbp = zeros(l,1);
mWr = zeros(l,m);   mbr = zeros(l,1);
mWq = zeros(l,m);   mbq = zeros(l,1);
mV = zeros(n,l);    mb = zeros(n,1);
% -------------------------------------------------------------------------

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
        bEnd = min(bStart+bSize-1, train_size);
        
        inputs = XTrain(bStart: bEnd);
        targets = YTrain(bStart: bEnd);
        
        [dWf,dRf,dbf,dWi,dRi,dbi,dWg,dRg,dbg,dWo,dRo,dbo, ...
            dWp,dbp,dWr,dbr,dWq,dbq, dV,db, h0, c0, loss] = ...
            lstm_highway(Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo, ...
            Wp,bp,Wr,br,Wq,bq,V,b, inputs, targets, h0, c0);

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
        
        mWp = mnt_rate*mWp - learning_rate*dWp;
        mWq = mnt_rate*mWq - learning_rate*dWq;
        mWr = mnt_rate*mWr - learning_rate*dWr;
        
        mbp = mnt_rate*mbp - learning_rate*dbp;
        mbq = mnt_rate*mbq - learning_rate*dbq;
        mbr = mnt_rate*mbr - learning_rate*dbr;
        
        mV = mnt_rate*mV - learning_rate*dV;
        mb = mnt_rate*mb - learning_rate*db;

        Wf = Wf + mWf;  Rf = Rf + mRf; bf = bf + mbf;
        Wi = Wi + mWi;  Ri = Ri + mRi; bi = bi + mbi;
        Wg = Wg + mWg;  Rg = Rg + mRg; bg = bg + mbg;
        Wo = Wo + mWo;  Ro = Ro + mRo; bo = bo + mbo;
        Wp = Wp + mWp;  bp = bp + mbp;
        Wq = Wq + mWq;  bq = bq + mbq;
        Wr = Wr + mWr;  br = br + mbr;
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
[hend, cend, yy] = lstm_highway_forward(...
    Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo, ...
    Wp,bp,Wr,br,Wq,bq,V,b, XTrain, h0, c0);

xx = 1:numel(XTrain)-1;
figure;
xxtrain = XTrain*mu + E;
yytrain = yy*mu + E;

plot(xx, xxtrain(2:end),'.-', xx, yytrain(1:end-1), '*-');
legend('input', 'pred');


pred = zeros(n, test_size);
for j = 1:test_size
    [hend, cend, pred(:,j)] = lstm_highway_forward(...
         Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo, ...
         Wp,bp,Wr,br,Wq,bq,V,b, XTest(:,j), hend, cend);
end

pred = pred*mu + E;
xx = 1:test_size;
%plot(xx, YTest, xx, pred, xx, (XTest*mu+E));
figure;
plot(xx, YTest,'.-', xx, pred, '*-');
legend('target', 'pred');

RMSE = sqrt(mean((YTest - pred).^2))
