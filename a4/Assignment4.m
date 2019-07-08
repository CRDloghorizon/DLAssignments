% DD2424 deeplearning 2019 assignment4
% train an vanilla RNN to synthesize English text character by character
% LogHorizon

%clear all; 
clc;

disp("1: run test part;");
disp("2: run training part:");

exercise = input("Input: ");

% load data and set two containers
book_data=LoadData();
C = unique(book_data);
k = numel(C);
char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');
for i= 1:k
    char_to_ind(C(i)) = i;
    ind_to_char(i) = C(i);
end

RNN.char_to_ind = char_to_ind;
RNN.ind_to_char = ind_to_char;
RNN.K = k;
RNN.N = size(book_data, 2);


if exercise == 1
    RNN.m = 5;
    RNN.seq_length = 25;
    RNN.sig = 0.01;
    RNN = InitRNN(RNN);
    X = zeros(RNN.K, 30);
    h0 = zeros(RNN.m, 1);
    % inject Xtr and Ytr
    for i = 1:30
        X(char_to_ind(book_data(i)), i) = 1;
    end
    Xtr = X(:,1:RNN.seq_length);
    Ytr = X(:,2:RNN.seq_length+1);
    grads = ComputeGradients(Xtr, Ytr, RNN, h0);
    num_grads = ComputeGradsNum(Xtr, Ytr, RNN, 1e-4, grads);
    for f = fieldnames(grads)'
       abserr.(f{1}) = max(max(grads.(f{1}) - num_grads.(f{1})));
       epserr.(f{1}) = abs(grads.(f{1}) - num_grads.(f{1}))./(max(eps,abs(grads.(f{1})) + abs(num_grads.(f{1}))));
       epserr.(f{1}) = max(max(epserr.(f{1})));
    end
    disp(abserr);
    disp(epserr);
end

if exercise == 2
    RNN.m = 100;
    RNN.seq_length = 25;
    RNN.sig = 0.01;
    RNN = InitRNN(RNN);
    param.n_epoch = 20;
    param.eta = 0.1;
    param.eps = 1e-10;
    X = zeros(RNN.K, RNN.N);
    for i = 1:RNN.N
        X(char_to_ind(book_data(i)), i) = 1;
    end
    [RNN, loss] = AdaGD(RNN, X, param);
    
    plot(loss.step, loss.trainloss, 'g');
    legend('smooth loss')
    xlabel('update step')
    ylabel('loss')
    print("f2.png", '-dpng');
    %pause;
    close gcf
end



function book_data = LoadData()
    book_fname = 'goblet_book.txt';
    fid = fopen(book_fname, 'r');
    book_data = fscanf(fid, '%c');
    fclose(fid);
end

function RNN = InitRNN(RNN)
    m = RNN.m;
    K = RNN.K;
    sig = RNN.sig;
    RNN.b = zeros(m, 1);
    RNN.c = zeros(K, 1);
    RNN.U = randn(m, K)*sig;                          
    RNN.W = randn(m, m)*sig;
    RNN.V = randn(K, m)*sig;
end

function Y = Synthesize(RNN, h0, x0, n)
% input h0: hidden m*1, x0: input K*1, n: final length
    x = x0;
    h = h0;
    Y = zeros(RNN.K, n);   
    for i = 1:n
        a = RNN.W * h + RNN.U * x + RNN.b; % m*1
        h = tanh(a);
        o = RNN.V * h + RNN.c; % k*1
        p = exp(o)./sum(exp(o));        
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a>0);
        ii = ixs(1);
        Y(ii, i) = 1;
        x = Y(:, i);
    end
end

function [P, H, J]= forward(X, Y, RNN, h0)
% X, Y are k*seq_length
    h = h0;
    n = RNN.seq_length;
    P = zeros(RNN.K, n);
    H = zeros(RNN.m, n);
    for i = 1:n
        x = X(:, i); % k*1
        a = RNN.W * h + RNN.U * x + RNN.b; % m*1
        h = tanh(a); % m*1
        o = RNN.V * h + RNN.c; % k*1
        p = exp(o)./sum(exp(o));
        H(:, i) = h; % m*n
        P(:, i) = p; % k*n
    end
    J = -sum(log(sum(Y .* P, 1)));
end

function J = ComputeLoss(X, Y, RNN, h0)
    [~, ~, J] = forward(X, Y, RNN, h0);
end

function grads = ComputeGradients(X, Y, RNN, h0)
    gradnames = {'b', 'c', 'U', 'W', 'V'};
    for i = 1:length(gradnames)
        grads.(gradnames{i}) = zeros(size(RNN.(gradnames{i})));
    end
    n = RNN.seq_length;    
    [P, H, ~] = forward(X, Y, RNN, h0);
    G = (P - Y)'; % n*k
    grads.c = sum(G)'; % k*1
    grads.V = G' * H'; % k*m
    grad_ht = G(n, :) * RNN.V; % 1*m h_tao
    % t-1 to 2 compute ht at w u b
    for i = n:-1:2
        grad_at = grad_ht * diag(1 - H(:,i).^2); %1*m a_t to a_2
        grads.W = grads.W + grad_at' * H(:,i-1)'; % m*m w_t to w_2
        grads.U = grads.U + grad_at' * X(:, i)'; % m*k u_t to u_2
        grads.b = grads.b + grad_at'; % m*1 b_t to b_2
        grad_ht = G(i-1, :) * RNN.V + grad_at * RNN.W; % 1*m h_(t-1) to h_1       
    end
    % t = 1 compute h0 a1 w1 u1 b1
    grad_at = grad_ht * diag(1 - H(:,1).^2);
    grads.W = grads.W + grad_at' * h0';
    grads.U = grads.U + grad_at' * X(:, 1)';
    grads.b = grads.b + grad_at';
    % avoid the exploding gradient problem
    for f = fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
end

function [RNN, loss] = AdaGD(RNN, X, param)
    fidt = fopen("text_result2.txt", 'w');

    n_epoch = param.n_epoch;
    eta = param.eta;
    eps = param.eps;
    n = RNN.seq_length;
    hprev = zeros(RNN.m, 1);
    gradnames = {'b', 'c', 'U', 'W', 'V'};
    for i = 1:length(gradnames)
        m.(gradnames{i}) = zeros(size(RNN.(gradnames{i})));
    end
    updatestep = 0;
    count = 0;
    for i=1:n_epoch
        e = 1;
        while (e+n < RNN.N)
            updatestep = updatestep + 1;
            Xtr = X(:, e:e+n-1);
            Ytr = X(:, e+1:e+n);
            e = e + n;
            grads = ComputeGradients(Xtr, Ytr, RNN, hprev);
            % AdaGrad SGD
            for f = fieldnames(grads)'
                m.(f{1}) = m.(f{1}) + grads.(f{1}).^2;
                RNN.(f{1}) = RNN.(f{1}) - eta * grads.(f{1})./sqrt(m.(f{1}) + eps);
            end
            [~, H, J]= forward(Xtr, Ytr, RNN, hprev);
            hprev = H(:,n);
            if updatestep==1
                smooth_loss = J;
            else
                smooth_loss = 0.999 * smooth_loss + 0.001 * J;
            end
            % record and print 1, 1000, 4000, per 10000th
            if updatestep == 1 || updatestep == 1000 || updatestep == 4000
                disp("iter = " + updatestep +", smooth_loss = "+smooth_loss);
                fprintf(fidt,'iter = %d, smooth_loss = %.4f\n',updatestep,smooth_loss);
                xin = X(:, e);
                yout = Synthesize(RNN, hprev, xin, 200); % k*n
                text(200) = ' ';
                for j = 1:200
                   text(j) = RNN.ind_to_char(find(yout(:, j)==1));
                end
                disp(text);
                fprintf(fidt,'%s\n',text);
            elseif mod(updatestep, 500) == 0
                count = count + 1;
                loss.trainloss(count) = smooth_loss;
                loss.step(count) = updatestep;
                % disp("iter = " + updatestep +", smooth_loss = "+smooth_loss);
                if mod(updatestep, 10000) == 0
                    disp("iter = " + updatestep +", smooth_loss = "+smooth_loss);
                    fprintf(fidt,'iter = %d, smooth_loss = %.4f\n',updatestep,smooth_loss);
                    xin = X(:, e);
                    yout = Synthesize(RNN, hprev, xin, 200); % k*n
                    text(200) = ' ';
                    for j = 1:200
                       text(j) = RNN.ind_to_char(find(yout(:, j)==1));
                    end
                    disp(text);
                    fprintf(fidt,'%s\n',text);
                end
            end
        end
    end
    xin = X(:, e);
    yout = Synthesize(RNN, hprev, xin, 1000); % k*n
    text(1000) = ' ';
    for j = 1:1000
       text(j) = RNN.ind_to_char(find(yout(:, j)==1));
    end
    disp(text);
    fprintf(fidt,'Final Text: \n %s\n',text);
    
    fclose(fidt);
end


function num_grads = ComputeGradsNum(X, Y, RNN, h, grads)
    for f = fieldnames(grads)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNumSlow(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end