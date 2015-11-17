function res = jk_linearNN_multi_v1(net, res_input, dzdy, varargin)
n = numel(net.layers);
if (nargin<=2) || isempty(dzdy)
    do_forward = true;
else
    do_forward = false;
end

opts.res = [] ;
opts.conserveMemory = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts = vl_argparse(opts, varargin);
res = res_input;

% forward
if do_forward
    for i=1:n
        layer = net.layers{i};
%        res(i).forwardtime = tic;
        switch layer.type
            case 'dot'
                res(i+1).x = jk_dot(res(i).x, layer.weight, layer.bias);
            case 'l2norm'
                res(i+1).x = jk_l2norm(res(i).x);
            case 'tanh'
                res(i+1).x = jk_tanh(res(i).x);
            otherwise
                error('Unknow layer type %s', layer.type);
        end
%        res(i).forwardtime = toc(res(i).forwardtime);
    end
else
    % backward
    res(n+1).dzdx = dzdy; 
    for i=n:-1:1
        layer = net.layers{i};
%        res(i).backwardtime = tic;
        switch layer.type
            case 'dot'
                [res(i).dzdx, res(i).dzdw, res(i).dzdb] = jk_dot(res(i).x, layer.weight, layer.bias, res(i+1).dzdx);
            case 'l2norm'
                [res(i).dzdx] = jk_l2norm(res(i).x, res(i+1).dzdx);
            case 'tanh'
                res(i).dzdx = jk_tanh(res(i).x, res(i+1).dzdx);
            otherwise
                error('Unknow layer type %s', layer.type);
        end
%        res(i).backwardtime = toc(res(i).backwardtime);
    end
end
