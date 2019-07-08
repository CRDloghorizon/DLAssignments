% DD2424 deeplearning 2019 assignment3
% train a ConvNet to predict the language of a surname from its spelling
% LogHorizon

%clear all; 
clc;

disp("1: run test part;");
disp("2: run training part:");
disp("3: run validation part");
disp("4: run speed up test part");
exercise = input("Input: ");

if exercise == 1
    % cheack convolution matrices
%     load('DebugInfo.mat');
%     [d, nlen] = size(X_input);
%     k = size(F, 2);
%     nf = 1;
%     MX = MakeMXMatrix(X_input, d, k, nf);
%     MF = MakeMFMatrix(F(:, :, 1), nlen);
%     s1 = MX * vecF(1:140);
%     s2 = MF * x_input;
%     %disp(s1-s2);
%     testS = zeros(4, 15);
%     for i = 1:4
%         MF = MakeMFMatrix(F(:, :, i), nlen);
%         testS(i, :) = MF * x_input;
%     end
%     vectestS = testS(:);
%     disp(vectestS - vecS);  % err around 1e-15
%     disp(testS - S);
    % check gradient
    %[X, Y, ys] = LoadName();
    load('loadname.mat', 'X', 'Y', 'ys');
    X = X(:, :, 1:10);
    Y = Y(:, 1:10);
    y = ys(1:10);
    % [d, nlen ,N] = size(X);
    d = size(X, 1);
    nlen = size(X, 2);
    N = size(X, 3);
    k = size(Y, 1);
    [k1, n1, k2, n2] = deal(5, 20, 3, 20);
    nlen1 = nlen-k1+1;
    nlen2 = nlen1-k2+1;
    %HE initialization sqrt(2/fan_in)
    sig1 = sqrt(2/(n1*nlen1));
    sig2 = sqrt(2/(n2*nlen2));
    sig3 = sqrt(2/k);
    ConvNet.F{1} = randn(d, k1, n1)*sig1;
    ConvNet.F{2} = randn(n1, k2, n2)*sig2;
    ConvNet.W = randn(k, n2*nlen2)*sig3;
    [gradF1, gradF2, gradW] = ComputeGradients(X, Y, ConvNet);
    Gs = NumericalGradient(X, Y, ConvNet, 1e-5);
    % absolutes difference are small (<=1e-9)
    abs_errw = max(max(gradW - Gs{3}));
    abs_errf1 = max(max(max(gradF1 - Gs{1})));
    abs_errf2 = max(max(max(gradF2 - Gs{2})));
    disp("W: " + abs_errw + ", f1: " + abs_errf1 + ", f2: " + abs_errf2);
    % W: 5.9741e-11, f1: 3.7766e-11, f2: 6.1147e-11
    % eps test
    err_w = abs(gradW - Gs{3})./(max(eps,abs(gradW) + abs(Gs{3})));
    err_w = max(max(err_w));
    err_f1 = abs(gradF1 - Gs{1})./(max(eps,abs(gradF1) + abs(Gs{1})));
    err_f1 = max(max(max(err_f1)));
    err_f2 = abs(gradF2 - Gs{2})./(max(eps,abs(gradF2) + abs(Gs{2})));
    err_f2 = max(max(max(err_f2)));
    disp("W: " + err_w + ", f1: " + err_f1 + ", f2: " + err_f2);
    % W: 1.8245e-06, f1: 7.8813e-07, f2: 2.5739e-07
end

if exercise == 2
    %[X, Y, ys] = LoadName();
    load('loadname.mat', 'X', 'Y', 'ys');
    validIndex = importdata('Validation_Inds.txt');
    validX = X(:,:,validIndex);
    validY = Y(:,validIndex);
    validy = ys(validIndex);
    trainIndex = 1:size(X, 3);
    trainIndex(validIndex) = -1;
    trainIndex = trainIndex(trainIndex>0);
    trainX = X(:,:,trainIndex);
    trainY = Y(:,trainIndex);
    trainy = ys(trainIndex);
    % load data finish, set ConvNet's parameters
    d = size(X, 1);
    nlen = size(X, 2);
    N = size(X, 3);
    k = size(Y, 1);
    [k1, n1, k2, n2] = deal(5, 20, 3, 20);
    nlen1 = nlen-k1+1;
    nlen2 = nlen1-k2+1;
    %HE initialization sqrt(2/fan_in)
    sig1 = sqrt(2/(n1*nlen1));
    sig2 = sqrt(2/(n2*nlen2));
    sig3 = sqrt(2/k);
    ConvNet.F{1} = randn(d, k1, n1)*sig1;
    ConvNet.F{2} = randn(n1, k2, n2)*sig2;
    ConvNet.W = randn(k, n2*nlen2)*sig3;
    % setset hyper-parameters for network (6)
    params.n_batch = 100;
    params.n_epoch = 50000;
    params.eta = 0.001;
    params.rho = 0.9;
    params.n_update = 500;
    params.fin_update = 20000;
    params.compensate = 1; % 0
    [ConvNet, trainloss, validationloss, trainacc, validacc, M] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, ConvNet, params);
    % plot validation loss
    step = (1:length(validationloss)).*params.n_update;
    plot(step, trainloss, 'g', step, validationloss, 'r');
    legend('training loss', 'validation loss')
    xlabel('update step')
    ylabel('loss')
    print("Result_Pics/f2_c.png", '-dpng');
    %pause;
    close gcf
    %plot confusion matrix
    imagesc(M);
    colorbar;
    print("Result_Pics/cm2_c.png", '-dpng');
    %pause;
    close gcf
    save("net1.mat", 'ConvNet', 'trainacc', 'validacc', 'M');
end

% class accuracy and name validation
if exercise == 3
    load('loadname.mat', 'X', 'Y', 'ys');
    validIndex = importdata('Validation_Inds.txt');
    validX = X(:,:,validIndex);
    validY = Y(:,validIndex);
    validy = ys(validIndex);
    load("net1.mat", 'ConvNet', 'trainacc', 'validacc', 'M');
    N = size(validX, 3);
    k = size(validY, 1);
    Xclass = {};
    Xclass{k} = [];
    for i = 1:N
        class = validy(i);
        Xclass{class}(end+1) = i;
    end
    for i = 1:k
        X_c = validX(:,:,Xclass{i});
        y_c = validy(Xclass{i});
        acc = ComputeAccuracy(X_c, y_c, ConvNet);
        disp("Class: "+ i + ", Acc is " + acc);
    end
    % test names
    name = {"zhu", "zhang", "song", "charlemagne", "johnson", "tsuyuzaki"};
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    [d, k, nlen] = deal(28, 18, 19);
    N = length(name);
    load('assignment3_names.mat', 'all_names');
    C = unique(cell2mat(all_names));
    for i = 1:d
        char_to_ind(C(i)) = i;
    end
    TX = zeros(d, nlen, N);
    for i = 1:N
        for j = 1:length(name{i})
            TX(char_to_ind(name{i}(j)), j, i) = 1;
        end
    end
    [P,~,~] = EvaluateClassifier(TX, ConvNet);
    [~, predict] = max(P);
    disp("predict is " + predict);
end

if exercise == 4
    load('loadname.mat', 'X', 'Y', 'ys');
    validIndex = importdata('Validation_Inds.txt');
    validX = X(:,:,validIndex);
    validY = Y(:,validIndex);
    validy = ys(validIndex);
    trainIndex = 1:size(X, 3);
    trainIndex(validIndex) = -1;
    trainIndex = trainIndex(trainIndex>0);
    trainX = X(:,:,trainIndex);
    trainY = Y(:,trainIndex);
    trainy = ys(trainIndex);
    % load data finish, set ConvNet's parameters
    d = size(X, 1);
    N = size(X, 3);
    nlen = size(X, 2);
    k = size(Y, 1);
    [k1, n1, k2, n2] = deal(5, 20, 3, 20);
    nlen1 = nlen-k1+1;
    nlen2 = nlen1-k2+1;
    %HE initialization sqrt(2/fan_in)
    sig1 = sqrt(2/(n1*nlen1));
    sig2 = sqrt(2/(n2*nlen2));
    sig3 = sqrt(2/k);
    ConvNet.F{1} = randn(d, k1, n1)*sig1;
    ConvNet.F{2} = randn(n1, k2, n2)*sig2;
    ConvNet.W = randn(k, n2*nlen2)*sig3;
    % setset hyper-parameters for network (6)
    params.n_batch = 100;
    params.n_epoch = 50000;
    params.eta = 0.001;
    params.rho = 0.9;
    params.n_update = 100;
    params.fin_update = 100;
    params.compensate = 1;
    setpre = 0; % 1 after changing n1 or k1
    % pre compute mx
    if setpre == 1
        NX = size(trainX, 3);
        MXpre = cell(1, NX);
        for i = 1:NX
            MXpre{i} = sparse(MakeMXMatrix(trainX(:,:,i), d, k1, n1));
        end
        save("MXprec.mat", 'MXpre');
    else
        load("MXprec.mat", 'MXpre');
    end
    %[ConvNet, trainloss, validationloss, trainacc, validacc, M] = MiniBatchGD_MX(trainX, trainY, trainy, validX, validY, validy, ConvNet, params, MXpre);
    % 18.134s
    [ConvNet, trainloss, validationloss, trainacc, validacc, M] = MiniBatchGD(trainX, trainY, trainy, validX, validY, validy, ConvNet, params);
    % 42.952s
    % save("net1.mat", 'ConvNet', 'trainacc', 'validacc', 'M');
    % plot validation loss
    step = (1:length(validationloss)).*params.n_update;
    plot(step, trainloss, 'g', step, validationloss, 'r');
    legend('training loss', 'validation loss')
    xlabel('update step')
    ylabel('loss')
    %print("Result_Pics/f1_nc.png", '-dpng');
    pause;
    close gcf
    %plot confusion matrix
    imagesc(M);
    colorbar;
    %print("Result_Pics/cm1_nc.png", '-dpng');
    pause;
    close gcf
    % acc plot
    plot(step, trainacc, 'g', step, validacc, 'r');
    legend({'training acc', 'validation acc'}, 'Location','southeast')
    xlabel('update step')
    ylabel('accuracy')
    %print("Result_Pics/f1_acc_nc.png", '-dpng');
    pause;
    close gcf
end

function [ys, all_names] = ExtractNames()
    data_fname = 'ascii_names.txt';
    fid = fopen(data_fname,'r');
    S = fscanf(fid,'%c');
    fclose(fid);
    names = strsplit(S, '\n');
    if length(names{end}) < 1        
        names(end) = [];
    end
    ys = zeros(length(names), 1);
    all_names = cell(1, length(names));
    for i=1:length(names)
        nn = strsplit(names{i}, ' ');
        l = str2num(nn{end});
        if length(nn) > 2
            name = strjoin(nn(1:end-1));
        else
            name = nn{1};
        end
        name = lower(name);
        ys(i) = l;
        all_names{i} = name;
    end
end


function [X, Y, ys] = LoadName()
    %[ys, all_names] = ExtractNames();
    load('assignment3_names.mat', 'ys', 'all_names');
    C = unique(cell2mat(all_names));
    d = numel(C);    
    N = size(ys, 1);
    n_len = max(cellfun('length', all_names));
    k = length(unique(ys));
    char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
    % 1->28, fill in the charactersas its keys and create an integer for its value
    for i = 1:d
        char_to_ind(C(i)) = i;
    end
    X = zeros(d, n_len, N);
    for i = 1:N
        for j = 1:length(all_names{i})
            X(char_to_ind(all_names{i}(j)), j, i) = 1;
        end
    end
    Y = zeros(k, N);
    for i = 1:N
        Y(ys(i), i) = 1;
    end
    %save("loadname.mat", 'X', 'Y', 'ys');
end

function MF = MakeMFMatrix(F, nlen)
	%F d*k*nf, X d*nlen*N
    [d, k, nf] = size(F);
    MF = zeros((nlen-k+1)*nf, nlen*d);
    vectorF = zeros(nf, d*k); % nf * (Vec(d*k))
    for i = 1:nf
		ipf = F(:, :, i);
		vectorF(i, :) = ipf(:)';
    end
    for i = 1:(nlen-k+1)
        MF((nf*(i-1)+1):nf*i, (d*(i-1)+1):(d*(i-1)+d*k)) = vectorF;
    end
end

function MX = MakeMXMatrix(x_input, d, k, nf)
    %  x_input d*nlen (N=nf), MX: I(nf) * vec(X:,1:2)...
    nlen = size(x_input, 2);
    MX = zeros((nlen-k+1)*nf, d*k*nf);
    vectorX = x_input(:);
    for i = 1:(nlen-k+1)
        for j = 1:nf
            MX((i-1)*nf+j, (k*d*(j-1)+1):k*d*j) = vectorX(((i-1)*d+1):((i-1)*d+k*d));
        end
    end
end

function [P, X1, X2]= EvaluateClassifier(X, ConvNet)
    % layer 1: X(1) = max(MF1*X,0) (n1*nlen1)*N
    % layer 2: X(2) = max(MF2*X(1),0) (n2*nlen2)*N
    % fully connected layer S = W*X(2) W:k*(n2*nlen2); P =softmax(S) k*N
    F1 = ConvNet.F{1};
    F2 = ConvNet.F{2};
    W = ConvNet.W;
    d = size(X, 1);
    nlen = size(X, 2);
    N = size(X, 3);
    k1 = size(F1, 2);
    k2 = size(F2, 2);
    nlen1 = nlen-k1+1;
    nlen2 = nlen1-k2+1;
    MF1 = MakeMFMatrix(F1, nlen);
    MF2 = MakeMFMatrix(F2, nlen1);
    % go through layers
    xin = reshape(X, [d*nlen, N]);
    S1 = MF1 * xin;
    X1 = max(S1, 0);
    %X1_out = reshape(X1, [n1, nlen1, N]);
    S2 = MF2 * X1;
    X2 = max(S2, 0);
    %X2_out = reshape(X2, [n2, nlen2, N]);
    S = W * X2;
    P = exp(S)./(double(ones(size(W, 1)))*exp(S));
end

function J = Compute_loss(X, Y, ConvNet)
    [P, ~, ~]= EvaluateClassifier(X, ConvNet);
%     N = size(X, 3);
%     J = 0;
%     for i = 1:N
%         J = J - log(Y(:,i)'*P(:,i));
%     end
%     J = J / N;
    J = mean(-log(sum(Y .* P, 1)));
end

function acc = ComputeAccuracy(X, y, ConvNet)
    N = size(X, 3);
    [P, ~, ~]= EvaluateClassifier(X, ConvNet);
    [~, predict] = max(P);
    predict = predict';
    correct = find(y == predict);
    acc = length(correct)/N;
end

function [gradF1, gradF2, gradW] = ComputeGradients(X, Y, ConvNet)
    %  G,P,Y: k*N;  X:d*nlen*N;  W:k*(n2*nlen2)
    %  F1: d*k1*n1;  F2: n1*k2*n2;
    %  X1: (n1*nlen1)*N;  X2: (n2*nlen2)*N
    F1 = ConvNet.F{1};
    F2 = ConvNet.F{2};
    W = ConvNet.W;
    gradF1 = zeros(size(F1));
    gradF2 = zeros(size(F2));
    %gradW = zeros(size(W));
    [P, X1, X2]= EvaluateClassifier(X, ConvNet);
    [d, k1, n1] = size(F1);
    [~, k2, n2] = size(F2);
    N = size(X, 3);
    nlen1 = size(X1, 1)/n1;
    X1_out = reshape(X1, [n1, nlen1, N]);

    G = P - Y;
    gradW = (G * X2')/N;
    
    G = W' * G;
    X2(X2 > 0) = 1;
    G = G .* X2; % (n2*nlen2)*N
    for j = 1:N
        gj = G(:, j); % (n2*nlen2) * 1
        xj = X1_out(:,:,j); % should be n1 * nlen1 * 1 for MakeMX
        MX = MakeMXMatrix(xj, n1, k2, n2); % (nlen2*n2) * (n1*k2*n2)
        v = gj' * MX; % 1 * (n1*k2*n2)
        v = reshape(v, [n1, k2, n2]); % reshape to n1*k2*n2
        gradF2 = gradF2 + v./N; 
    end
    
    MF2 = MakeMFMatrix(F2, nlen1); % (nlen2*n2) * (n1*nlen1)
    G = MF2' * G; % (n1*nlen1) * N
    X1(X1 > 0) = 1;
    G = G .* X1; % (n1*nlen1)*N
    for j = 1:N
        gj = G(:, j); % (n1*nlen1) * 1
        xj = X(:,:,j); % d * nlen * 1
        MX = MakeMXMatrix(xj, d, k1, n1); % (nlen1*n1) * (d*k1*n1)
        v = gj' * MX; % 1 * (d*k1*n1)
        v = reshape(v, [d, k1, n1]); % reshape to d*k1*n1
        gradF1 = gradF1 + v./N; 
    end
   
end

function [gradF1, gradF2, gradW] = ComputeGradients_MX(X, Y, ConvNet, MXpre)
    %  G,P,Y: k*N;  X:d*nlen*N;  W:k*(n2*nlen2)
    %  F1: d*k1*n1;  F2: n1*k2*n2;
    %  X1: (n1*nlen1)*N;  X2: (n2*nlen2)*N
    F1 = ConvNet.F{1};
    F2 = ConvNet.F{2};
    W = ConvNet.W;
    gradF1 = zeros(size(F1));
    gradF2 = zeros(size(F2));
    %gradW = zeros(size(W));
    [P, X1, X2]= EvaluateClassifier(X, ConvNet);
    [d, k1, n1] = size(F1);
    [~, k2, n2] = size(F2);
    N = size(X, 3);
    nlen1 = size(X1, 1)/n1;
    X1_out = reshape(X1, [n1, nlen1, N]);

    G = P - Y;
    gradW = (G * X2')/N;
    
    G = W' * G;
    X2(X2 > 0) = 1;
    G = G .* X2; % (n2*nlen2)*N
    for j = 1:N
        gj = G(:, j); % (n2*nlen2) * 1
        xj = X1_out(:,:,j); % should be n1 * nlen1 * 1 for MakeMX
        MX = MakeMXMatrix(xj, n1, k2, n2); % (nlen2*n2) * (n1*k2*n2)
        v = gj' * MX; % 1 * (n1*k2*n2)
        v = reshape(v, [n1, k2, n2]); % reshape to n1*k2*n2
        gradF2 = gradF2 + v./N; 
    end
    
    MF2 = MakeMFMatrix(F2, nlen1); % (nlen2*n2) * (n1*nlen1)
    G = MF2' * G; % (n1*nlen1) * N
    X1(X1 > 0) = 1;
    G = G .* X1; % (n1*nlen1)*N
    for j = 1:N
        gj = G(:, j); % (n1*nlen1) * 1
%         xj = X(:,:,j); % d * nlen * 1
%         MX = MakeMXMatrix(xj, d, k1, n1); % (nlen1*n1) * (d*k1*n1)
        MX = MXpre{j};
        v = gj' * MX; % 1 * (d*k1*n1)
        v = reshape(v, [d, k1, n1]); % reshape to d*k1*n1
        gradF1 = gradF1 + v./N; 
    end
   
end


% second method for Compensating for unbalanced training data
function index = Balance(X, y, k)
    N = size(X, 3);
    Xclass = {};
    Xclass{k} = [];
    % X index <-> class, find the smallest class
    for i = 1:N
        class = y(i);
        Xclass{class}(end+1) = i;
    end
    minclass = min(cellfun('length', Xclass));
    index = zeros(minclass*k, 1);
    % for each class select minclass index
    for i = 1:k
        temp = randperm(length(Xclass{i}));
        index((i-1)*minclass+1:i*minclass) = Xclass{i}(temp(1:minclass));
    end
end

function [ConvNet, trainloss, validationloss, trainacc, validacc, M] = MiniBatchGD(X, Y, y, VX, VY, vy, ConvNet, GDparams)
    n_batch = GDparams.n_batch;
    n_epoch = GDparams.n_epoch;
    eta = GDparams.eta;
    rho = GDparams.rho;
    n_update = GDparams.n_update;
    fin_update = GDparams.fin_update;
    k = size(Y, 1);
    M = zeros(k, k);
    momentumW = zeros(size(ConvNet.W));
    momentumF1 = zeros(size(ConvNet.F{1}));
    momentumF2 = zeros(size(ConvNet.F{2}));
    trainloss = zeros(fin_update/n_update, 1);
    validationloss = zeros(fin_update/n_update, 1);
    trainacc = zeros(fin_update/n_update, 1);
    validacc = zeros(fin_update/n_update, 1);
    count = 0;
    for i = 1:n_epoch
        if GDparams.compensate == 1
            index = Balance(X, y, k);
            indexlen = length(index);
            shuffleindex = index(randperm(indexlen));
            Xcom = X(:, :, shuffleindex);
            Ycom = Y(:, shuffleindex);
        else
            Xcom = X;
            Ycom = Y;
        end
        N = size(Xcom, 3);
        for j = 1:(N/n_batch)
            count = count + 1;
            j_start = (j - 1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = Xcom(:, :, j_start:j_end);
            Ybatch = Ycom(:, j_start:j_end);
            [gradF1, gradF2, gradW] = ComputeGradients(Xbatch, Ybatch, ConvNet);
            % v(t+1) = rho*v(t)+eta*gradx(t) update vector
            % gradx(t+1) = gradx(t) - v(t+1)
            momentumW = rho*momentumW + eta*gradW;
            momentumF1 = rho*momentumF1 + eta*gradF1;
            momentumF2 = rho*momentumF2 + eta*gradF2;
            ConvNet.W = ConvNet.W - momentumW;
            ConvNet.F{1} = ConvNet.F{1} - momentumF1;
            ConvNet.F{2} = ConvNet.F{2} - momentumF2;
            
            if (mod(count, n_update)==0)
                c = count/n_update;
                trainacc(c) = ComputeAccuracy(X, y, ConvNet);
                validacc(c) = ComputeAccuracy(VX, vy, ConvNet);
                trainloss(c) = Compute_loss(X, Y, ConvNet);
                validationloss(c) = Compute_loss(VX, VY, ConvNet);
            end
            % end iteration after final update and record confusion matrix
            if count == fin_update
                [P,~,~]= EvaluateClassifier(VX, ConvNet);
                for f = 1:size(P,2)
                    [~,predict] = max(P(:,f));
                    if vy(f)~=predict
                        M(vy(f), predict) = M(vy(f), predict) + 1;
                    end
                end
                return;
            end
        end    
    end
    
end


function [ConvNet, trainloss, validationloss, trainacc, validacc, M] = MiniBatchGD_MX(X, Y, y, VX, VY, vy, ConvNet, GDparams, MXpre)
    n_batch = GDparams.n_batch;
    n_epoch = GDparams.n_epoch;
    eta = GDparams.eta;
    rho = GDparams.rho;
    n_update = GDparams.n_update;
    fin_update = GDparams.fin_update;
    k = size(Y, 1);
    M = zeros(k, k);
    momentumW = zeros(size(ConvNet.W));
    momentumF1 = zeros(size(ConvNet.F{1}));
    momentumF2 = zeros(size(ConvNet.F{2}));
    trainloss = zeros(fin_update/n_update, 1);
    validationloss = zeros(fin_update/n_update, 1);
    trainacc = zeros(fin_update/n_update, 1);
    validacc = zeros(fin_update/n_update, 1);
    count = 0;
    for i = 1:n_epoch
        if GDparams.compensate == 1
            index = Balance(X, y, k);
            indexlen = length(index);
            shuffleindex = index(randperm(indexlen));
            Xcom = X(:, :, shuffleindex);
            Ycom = Y(:, shuffleindex);
            MXcom = MXpre(shuffleindex);
        else
            Xcom = X;
            Ycom = Y;
            MXcom = MXpre;
        end
        N = size(Xcom, 3);
        for j = 1:(N/n_batch)
            count = count + 1;
            j_start = (j - 1)*n_batch + 1;
            j_end = j*n_batch;
            Xbatch = Xcom(:, :, j_start:j_end);
            Ybatch = Ycom(:, j_start:j_end);
            [gradF1, gradF2, gradW] = ComputeGradients_MX(Xbatch, Ybatch, ConvNet, MXcom);
            % v(t+1) = rho*v(t)+eta*gradx(t) update vector
            % gradx(t+1) = gradx(t) - v(t+1)
            momentumW = rho*momentumW + eta*gradW;
            momentumF1 = rho*momentumF1 + eta*gradF1;
            momentumF2 = rho*momentumF2 + eta*gradF2;
            ConvNet.W = ConvNet.W - momentumW;
            ConvNet.F{1} = ConvNet.F{1} - momentumF1;
            ConvNet.F{2} = ConvNet.F{2} - momentumF2;
            
            if (mod(count, n_update)==0)
                c = count/n_update;
                trainacc(c) = ComputeAccuracy(X, y, ConvNet);
                validacc(c) = ComputeAccuracy(VX, vy, ConvNet);
                trainloss(c) = Compute_loss(X, Y, ConvNet);
                validationloss(c) = Compute_loss(VX, VY, ConvNet);
            end
            % end iteration after final update and record confusion matrix
            if count == fin_update
                [P,~,~]= EvaluateClassifier(VX, ConvNet);
                for f = 1:size(P,2)
                    [~,predict] = max(P(:,f));
                    if vy(f)~=predict
                        M(vy(f), predict) = M(vy(f), predict) + 1;
                    end
                end
                return;
            end
        end    
    end
    
end


function Gs = NumericalGradient(X_inputs, Ys, ConvNet, h)
    try_ConvNet = ConvNet;
    Gs = cell(length(ConvNet.F)+1, 1);

    for l=1:length(ConvNet.F)
        try_ConvNet.F{l} = ConvNet.F{l};

        Gs{l} = zeros(size(ConvNet.F{l}));
        nf = size(ConvNet.F{l},  3);

        for i = 1:nf        
            try_ConvNet.F{l} = ConvNet.F{l};
            F_try = squeeze(ConvNet.F{l}(:, :, i));
            G = zeros(numel(F_try), 1);

            for j=1:numel(F_try)
                F_try1 = F_try;
                F_try1(j) = F_try(j) - h;
                try_ConvNet.F{l}(:, :, i) = F_try1; 

                l1 = Compute_loss(X_inputs, Ys, try_ConvNet);

                F_try2 = F_try;
                F_try2(j) = F_try(j) + h;            

                try_ConvNet.F{l}(:, :, i) = F_try2;
                l2 = Compute_loss(X_inputs, Ys, try_ConvNet);            

                G(j) = (l2 - l1) / (2*h);
                try_ConvNet.F{l}(:, :, i) = F_try;
            end
            Gs{l}(:, :, i) = reshape(G, size(F_try));
        end
    end

    % compute the gradient for the fully connected layer
    W_try = ConvNet.W;
    G = zeros(numel(W_try), 1);
    for j=1:numel(W_try)
        W_try1 = W_try;
        W_try1(j) = W_try(j) - h;
        try_ConvNet.W = W_try1; 

        l1 = Compute_loss(X_inputs, Ys, try_ConvNet);

        W_try2 = W_try;
        W_try2(j) = W_try(j) + h;            

        try_ConvNet.W = W_try2;
        l2 = Compute_loss(X_inputs, Ys, try_ConvNet);            

        G(j) = (l2 - l1) / (2*h);
        try_ConvNet.W = W_try;
    end
    Gs{end} = reshape(G, size(W_try));
end