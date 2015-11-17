% Demo for VIPeR dataset
clear; clc;
curName = mfilename('fullpath');
[demoDir, name, ext] = fileparts(curName);
cd(demoDir);
addpath(genpath(fullfile(demoDir, '../')));
addpath(genpath(fullfile(demoDir, '../../code/')));
resDir = fullfile(demoDir, 'result');
mkdir(resDir);
matDir = fullfile(demoDir, '..\..\mat\viper');
load(fullfile(matDir, 'partition_random.mat'));
dim1 = 600;
dim2 = 200;
LossType = {'auc', '01'}; % 2_order, 01

lossType = '2_order';
for trial = 1:10
trnSp = partition(trial).trnSp;
trnSg = partition(trial).trnSg;
tstSp = partition(trial).tstSp;
tstSg = partition(trial).tstSg;

feature = importdata(fullfile(demoDir, '..\..\cache\viper\feature\LOMO_viper.mat')); % 34*1264

% ==== 预处理：method 1: 仅对训练集的数据做PCA===========
FeatP = feature(:, 1:2:end);
FeatG = feature(:, 2:2:end);

FeatP_trn_ori = FeatP(:, trnSp);
FeatG_trn_ori = FeatG(:, trnSg);
FeatP_tst_ori = FeatP(:, tstSp);
FeatG_tst_ori = FeatG(:, tstSg);
FeatTrn_ori = [FeatP_trn_ori, FeatG_trn_ori];
meanTrn = mean(FeatTrn_ori, 2);

options.ReducedDim = dim1;
[W, ux] = myPCA(FeatTrn_ori', options);

clear feature FeatP FeatG FeatTrn;
FeatP_trn = W'*(FeatP_trn_ori - repmat(meanTrn, [1, size(FeatP_trn_ori, 2)]));
FeatG_trn = W'*(FeatG_trn_ori - repmat(meanTrn, [1, size(FeatG_trn_ori, 2)]));
FeatP_tst = W'*(FeatP_tst_ori - repmat(meanTrn, [1, size(FeatP_tst_ori, 2)]));
FeatG_tst = W'*(FeatG_tst_ori - repmat(meanTrn, [1, size(FeatG_tst_ori, 2)]));
% 把所有数据都放到Feat_trn里面去
Feat_trn = zeros(size(FeatP_trn, 1), size(FeatP_trn, 2)+size(FeatG_trn, 2));
Feat_trn(:, 1:2:end) = FeatP_trn;
Feat_trn(:, 2:2:end) = FeatG_trn;
% clear FeatP_trn FeatG_trn;

%-----------initialization--------------------
feature = Feat_trn;
sigma = cov(feature');
[uu,ss,vv] = svd(sigma);
w = uu(:,1:dim2);
clear feature;

% net architecture
net.layers = {};
net.layers{end+1} = struct('type', 'dot', ...
    'weight', w, ...
    'bias', zeros(1, dim2), ...
    'lr', 1);
net.layers{end+1} = struct('type', 'tanh');
net.layers{end+1} = struct('type', 'l2norm');
% net.layers{end+1} = struct('type', 'l2norm');

nLayers = numel(net.layers);
res_template = struct('x', cell(1, nLayers+1), ...
             'dzdw', cell(1, nLayers+1), ...
             'dzdb', cell(1, nLayers+1), ...
             'dzdx', cell(1, nLayers+1), ...
             'aux', cell(1, nLayers+1));

% sum of gradient increment
sum_grad_template = struct('dzdw', cell(1, nLayers+1), ...
    'dzdb', cell(1, nLayers+1));
sum_grad_template(1).dzdw = zeros(dim1, dim2); % dot_group
sum_grad_template(1).dzdb = zeros(1, dim2);

% training
info.train.loss = [];
info.train.regu = [];
info.train.rank1 = []; % 这个以cmc rank-1作为error的指标么？
info.val.loss = [];
info.val.regu = [];
info.val.rank1 = [];

% % training options
opts.epochs = 50;
opts.learningRate = linspace(50*1e-4, 50*1e-4, 50);
opts.momentum = 0.9;
% ##########important parameters ############
C = 1;
lamda = 1*1e-2;
gamma = 0;
nSamples = size(Feat_trn, 2);

for epoch=1:300
    info.train.loss(end+1) = 0;
    info.train.regu(end+1) = 0;
    info.train.rank1(end+1) = 0;
    info.val.loss(end+1) = 0;
    info.val.rank1(end+1) = 0;
    info.val.regu(end+1) = 0;

    ksi = 0; regu = 0;
    trnNum = 0;
    disp('====================================');
    
    % embedding images
    result = cell(1, nSamples); 
    FP = zeros(dim2, nSamples);
    sum_grad_extra = sum_grad_template; % 每个epoch更新一次。所以每个epoch清一次0
    sum_grad_intra = sum_grad_template;
    tic;
    fprintf('embedding each image into target space');
    for i=1:nSamples
        if mod(i,50)==0
            fprintf('.');
        end
        x = Feat_trn(:,i);
        res = res_template; 
        res(1).x = x'; 
        res = jk_linearNN_multi_v1(net, res);
        result{i} = res;
        fp = res(end).x;
        FP(:,i) = fp';
    end
    fprintf('\n');
    
    % calculate dist between all images
    dist_trn = zeros(nSamples, nSamples);
    for i=1:nSamples
        for j=1:nSamples
            fp = FP(:, i);
            fg = FP(:, j);
            dd = sum((fp - fg).^2);
            dist_trn(i,j) = dd;
        end
    end
    
    % generate probe, positive and negative lists
    id_trn = [1:numel(trnSp); 1:numel(trnSp)]; id_trn = id_trn(:)';
    List = genList(id_trn);
    Y_star = [];
    fprintf('find most viloated constraint for each list');
    for i=1:size(List,1)
        if mod(i, 50)==0
            fprintf('.');
        end
        list = List(i,:);
        prb = list(1); pos = list(2); neg = list(3:end);
        sim_pos = -dist_trn(prb, pos);
        sim_neg = -dist_trn(prb, neg);
        
        % find most viloated constraint
        y_gt = ones(1, numel(neg));
        [slack, y_star] = findMostViolatedMvsM_fast(sim_pos, sim_neg, y_gt, lossType);
        
        % calculate loss
        ksi = ksi + slack;
        Y_star = cat(1, Y_star, y_star);
    end
    fprintf('\n');
    % for negative samples that y_j = -1, calculate the gradients of
    % loss layer
    fprintf('BP loss layer');
    DzDfp = zeros(size(FP));
    for i=1:size(Y_star, 1)
        if mod(i,50)==0
            fprintf('.');
        end
        list = List(i,:);
        prb = list(1); pos = list(2); neg = list(3:end);
        
        y_star = Y_star(i,:);
        y_gt = ones(1, numel(neg));
        
        PosNum = numel(pos); NegNum = numel(neg);
        A = PosNum*NegNum;
        
        for j=1:numel(neg)
            % for positive image
            dzdp_pos = (2/A)*(y_star(j) - y_gt(j))*FP(:,prb);
            DzDfp(:,pos) = DzDfp(:,pos) + dzdp_pos;
            
            % for negative image
            dzdp_neg = (-2/A)*(y_star(j) - y_gt(j))*FP(:,prb);
            DzDfp(:,neg(j)) = DzDfp(:,neg(j)) + dzdp_neg;
            
            % for probe image
            dzdp = (2/A)*(y_star(j) - y_gt(j))*(FP(:,pos) - FP(:,neg(j)));
            DzDfp(:,prb) = DzDfp(:,prb) + dzdp;
        end
    end
    fprintf('\n');
    
    % BP embedding layer
    fprintf('BP embedding layer');
    for i=1:size(DzDfp, 2)
        if mod(i,50)==0
            fprintf('.');
        end
        dzdp = DzDfp(:,i);
        res = result{i};
        res_backward = jk_linearNN_multi_v1(net, res, dzdp');
        % accumulate gradients for each result
        for n=1:numel(net.layers)
            layer = net.layers{n};
            switch layer.type
                case 'dot'
                    sum_grad_extra(n).dzdw = sum_grad_extra(n).dzdw + res_backward(n).dzdw;
                    sum_grad_extra(n).dzdb = sum_grad_extra(n).dzdb + res_backward(n).dzdb;
                otherwise
                    continue;
            end
        end % for each layer in the network
    end
    fprintf('\n');
    trainTime = toc;
    fprintf('training time is %.2f s', trainTime);

    % updata parameters per probe image
    learningRate_tmp = opts.learningRate(min(epoch, numel(opts.learningRate)));
    lamda_tmp = lamda(min(epoch, numel(lamda)));
    for n=1:numel(net.layers)
        layer = net.layers{n};
        switch layer.type
            case 'dot'
                weight = layer.weight;
                bias = layer.bias;
                layer.weight = layer.weight - learningRate_tmp * layer.lr * sum_grad_extra(n).dzdw ...
                    - learningRate_tmp * layer.lr * lamda_tmp * layer.weight;

                layer.bias = layer.bias - learningRate_tmp * layer.lr * sum_grad_extra(n).dzdb ...
                    - learningRate_tmp * layer.lr * lamda_tmp * layer.bias;
                net.layers{n} = layer;
            otherwise
                continue;
        end
    end
 
    info.train.loss(end) = ksi/nSamples;
    info.train.regu(end) = regu/nSamples;
%     info.train.loss(end) = ksi + regu;
    fprintf('\n');
    fprintf('***********trial %d, epoch: %d,\t training loss term: %.4f\n', trial, epoch, info.train.loss(end));
    fprintf('***********trial %d, epoch: %d,\t training regularization term: %.4f\n', trial, epoch, info.train.regu(end));

    %% test loss and cmc rank1
    disp('====================================');
    clear dist_tst cmc_tst dist_trn cmc_trn ksi ;
    dist_tst = zeros(numel(tstSp), numel(tstSg));
    FP_tst = zeros(dim2, size(FeatP_tst, 2));
    FG_tst = zeros(dim2, size(FeatG_tst, 2));
    ksi = 0; regu = 0;
    tic;
    % calculate feature map of probe set and gallery set in testing set
    for i=1:size(FeatP_tst, 2)
        x = FeatP_tst(:, i);
        res = res_template;
        res(1).x = x';
        res = jk_linearNN_multi_v1(net, res);
        fp = res(end).x';

        FP_tst(:,i) = fp';
    end
    for i=1:size(FeatG_tst, 2)
        x = FeatG_tst(:,i);
        res = res_template;
        res(1).x = x';
        res = jk_linearNN_multi_v1(net, res);
        fp = res(end).x';

        FG_tst(:,i) = fp';
    end

    % calculate dist 
    for i=1:size(FeatP_tst, 2)
        for j=1:size(FeatG_tst, 2)
            fp = FP_tst(:,i);
            fg = FG_tst(:,j);
            dd = sum((fp - fg).^2);
            dist_tst(i,j) = dd;
        end
    end

    % for each probe image, calculate the slack
    for i=1:size(FeatP_tst, 2)
        if mod(i, 10) == 0
            fprintf('.');
        end
        index_pos = i;
        index_neg = setdiff(1:size(FeatP_tst, 2), index_pos);

        Sim_pos = -dist_tst(i, index_pos);
        Sim_neg = -dist_tst(i, index_neg);

        % calculate slack
        y_gt = ones(numel(index_pos), numel(index_neg));
        [slack, y_star] = findMostViolatedMvsM_fast(Sim_pos, Sim_neg, y_gt, lossType);

        % calculate loss
        ksi = ksi + slack;
    end
    info.val.loss(end) = ksi/size(FeatP_tst, 2);
    info.val.regu(end) = regu/size(FeatP_tst, 2);
    cmc_tst = evaluate_cmc(-dist_tst)/numel(tstSp);
    tstTime = toc;

    fprintf('\n');
    fprintf('testing time is %.2f s\n', tstTime);
    fprintf('***********trial %d, epoch: %d, \t testing loss term: %.4f\n', trial, epoch, info.val.loss(end));
    fprintf('***********trial %d, epoch: %d, \t testing regularization term: %.4f\n',trial, epoch, info.val.regu(end));


    %% tranSet: cmc rank1
    dist_trn = zeros(numel(trnSp), numel(trnSg));
    disp('====================================');
    FP_trn = zeros(dim2, size(FeatP_trn, 2));
    FG_trn = zeros(dim2, size(FeatG_trn, 2));
    for i=1:size(FeatP_trn, 2)
        x = FeatP_trn(:,i);
        res = res_template; 
        res(1).x = x'; 
        res = jk_linearNN_multi_v1(net, res);
        fp = res(end).x;
        FP_trn(:, i) = fp';
    end
    for i=1:size(FeatG_trn, 2)
        x = FeatG_trn(:, i);
        res = res_template;
        res(1).x = x';
        res = jk_linearNN_multi_v1(net, res);
        fp = res(end).x;
        FG_trn(:, i) = fp';
    end
    for i=1:size(FP_trn, 2)
        if mod(i, 10) == 0
            fprintf('.');
        end
        for j=1:size(FG_trn, 2)
            fp = FP_trn(:, i);
            fg = FG_trn(:, j);
            dd = sum((fp - fg).^2);
            dist_trn(i,j) = dd;
        end
    end
    cmc_trn = evaluate_cmc(-dist_trn)/numel(trnSp);

    % plot result
    info.train.rank1(end) = cmc_trn(1);
    info.val.rank1(end) = cmc_tst(1);

    fprintf('\n');
    fprintf('*****loss %s, VIPeR: training set cmc *****\n', lossType);
    fprintf('rank1\t\trank5\t\trank10\t\trank15 \t\t rank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
        100*cmc_trn(1), 100*cmc_trn(5), 100*cmc_trn(10), 100*cmc_trn(15), 100*cmc_trn(20));
    fprintf('*****loss %s, VIPeR: testing set cmc *****\n', lossType);
    fprintf('rank1\t\trank5\t\trank10\t\trank15\t\trank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
        100*cmc_tst(1), 100*cmc_tst(5), 100*cmc_tst(10), 100*cmc_tst(15), 100*cmc_tst(20));
    
    hh=figure(1);clf;

    subplot(1,2,1);
    semilogy(1:epoch, info.train.loss, 'k-*'); hold on;
    semilogy(1:epoch, info.val.loss, 'g-o');
    xlabel('epoch'); ylabel('loss'); h = legend('train', 'val'); grid on;
    set(h, 'color', 'none');
    title('loss of train and validation');


    subplot(1,2,2);
    plot(1:epoch, info.train.rank1, 'k-*'); hold on;
    plot(1:epoch, info.val.rank1, 'g-o');ylim([0 1]);
    xlabel('epoch'); ylabel('error rate'); h = legend('train', 'val'); grid on;
    set(h, 'color', 'none');
    title('error');
    drawnow;
    saveas(hh, fullfile(resDir, [lossType '_viper_' num2str(trial) '.fig']), 'fig');
    if mod(epoch, 5) == 0
        save(fullfile(resDir, [lossType '_viper_' num2str(epoch) '_' num2str(trial) '.mat']), 'net', 'cmc_tst');
    end
    pause(1);
end
end
