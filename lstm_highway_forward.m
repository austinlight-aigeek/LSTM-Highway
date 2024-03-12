function [hend, cend, y] = lstm_highway_forward(...
    Wf,Rf,bf,Wi,Ri,bi,Wg,Rg,bg,Wo,Ro,bo,Wp,bp,Wr,br,Wq,bq,V,b, inputs, h0, c0)

T = size(inputs, 2);        % length of LSTM network
D = size(V, 1);   % number of output nodes

y = zeros(D, T);

ht = h0;
ct = c0;

for j = 1:T
    xt = inputs(:, j);
    
    ft = logsig(Wf*xt + Rf*ht + bf);
    it = logsig(Wi*xt + Ri*ht + bi);
    gt = tanh(Wg*xt + Rg*ht + bg);
    ot = logsig(Wo*xt + Ro*ht + bo);
    
    st = ft.*ct + it.*gt;
    pt = logsig(Wp*st + bp);
    rt = tanh(Wr*st + br);
    qt = logsig(Wq*st + bq);
    
    ct = st.*pt + rt.*qt;
    ht = ot.*tanh(ct);
    
    yt = V*ht + b;
    
    y(:,j) = yt;
end

hend = ht;
cend = ct;

end
