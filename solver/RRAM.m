function [xCur, xCurCost, info, options] = RRAM(problem, x0, options)
% Implementation of RRAM
%
% function [x, cost, info, options] = RRAM(problem)
% function [x, cost, info, options] = RRAM(problem, x0)
% function [x, cost, info, options] = RRAM(problem, x0, options)
% function [x, cost, info, options] = RRAM(problem, [], options)


    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the user
        % seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
           ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end

    
    % Local defaults for the program
    localdefaults.minstepsize = 1e-10; 
    localdefaults.maxiter = 500;
    localdefaults.tolgradnorm = 1e-8;
    localdefaults.stepsize = 0.1;

    localdefaults.memory = 5; % which is also the update frequency
    localdefaults.reg_lambda = 1e-8; 
    c1 = 1e-7;
    

    %%
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);

    % check averaging scheme been provided 
    if ~isfield(options, 'average')
        error('No averaging scheme provided. RiemNA cannot be applied.')
    end
    %averagefn = options.average;
    
    mu = options.mu;
%     p1 = options.p1;
%     p2 = options.p2;
%     eta1 = options.eta1;
%     eta2 = options.eta2;
%     c = options.c;
    stepsize = options.stepsize;
    beta = stepsize;
    
    % To make sure memory in range [2, Inf)
    % how many residual stored, it needs to be larger than 1, otherwise it
    % will be the same as RGD.
    options.memory = max(options.memory, 2); 
    if options.memory == Inf
        if isinf(options.maxiter)
            options.memory = 10000;
            warning('RRAM:memory', ['options.memory and options.maxiter' ...
              ' are both Inf; options.memory has been changed to 10000.']);
        else
            options.memory = options.maxiter;
        end
    end
    
    M = problem.M;
        
    % __________Initialization of variables______________
    % Create a random starting point if no starting point is provided.
    if ~exist('x0', 'var')|| isempty(x0)
        xCur = M.rand();
    else
        xCur = x0;
    end

    % Query the cost function and its gradient
    [newcost, newgrad] = getCostGrad(problem, xCur);
    newgradnorm = problem.M.norm(xCur, newgrad);
    xCurCost = newcost;
    xCurGradNorm = newgradnorm;

    % store the residual and function value and R matrix
    delta_rHistory = cell(1, options.memory); 
    delta_xHistory = cell(1, options.memory);
    delta_xHistory{1} = zeros(size(xCur));

    % preallocate
    info(min(10000, options.maxiter+1)).iter = [];
    info(min(10000, options.maxiter+1)).cost = [];
    info(min(10000, options.maxiter+1)).stepsize = [];
    info(min(10000, options.maxiter+1)).time = [];
    info(min(10000, options.maxiter+1)).gradnorm = [];
    info(min(10000, options.maxiter+1)).hooked = [];
    

    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t    grad. norm\n');
    end
    
    %% AA steps
    iter = 0; % counter k in the paper
    % performing one step of RGD
%     desc_dir = problem.M.lincomb(xCur, -1, newgrad);
%     curstep = problem.M.lincomb(xCur, stepsize, desc_dir);
    curstep = -stepsize * newgrad;
    newx = problem.M.retr(xCur, curstep);
    % Query the cost function and its gradient
    [newcost, newgrad] = getCostGrad(problem, newx);
    newgradnorm = problem.M.norm(newx, newgrad);

    % start timing
    timetic_riemna = tic();
    % Save stats in a struct array info.
    stats = savestats();
    %keyboard;
    info(iter+1) = stats;
    info = info(1:iter+1);
    iter = 1;
    % Start iterating until stopping criterion triggers.
    while true
        m = min(options.memory, iter);
        curr_idx = mod(iter, options.memory);
        if curr_idx == 0
            curr_idx = m;
        end
        pre_idx = curr_idx - 1;
        if pre_idx == 0
            pre_idx = m;
        end
        
        % update current pt
        xPre = xCur;
        r_pre = -newgrad;
        xCur = newx;
        xCurGradient = newgrad;
        xCurGradNorm = newgradnorm;
        xCurCost = newcost;
%         desc_dir = problem.M.lincomb(xCur, -1, xCurGradient);
%         curstep = problem.M.lincomb(xCur, stepsize, desc_dir);
        curstep = -stepsize * xCurGradient;
        r_curr = -newgrad;
        
%         rHistory{mod(iter, options.memory) + 1} = curstep;
%         xHistory{mod(iter, options.memory) + 1} = xCur;
        

        % Display iteration information.
        if options.verbosity >= 2
            fprintf('%5d\t%+.16e\t%.8e\n', iter, xCurCost, xCurGradNorm);
        end

        %% stopping criteria
        % Run standard stopping criterion checks.
        [stop, reason] = stoppingcriterion(problem, xCur, options, info, iter);

        % If none triggered, run specific stopping criterion check.
        if ~stop && stats.stepsize < options.minstepsize
            stop = true;
            reason = sprintf(['Last stepsize smaller than minimum '  ...
                              'allowed; options.minstepsize = %g.'], ...
                              options.minstepsize);
        end
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end
        
        % update the regularization lambda_k
        % xCurresNorm = M.norm(xCur, r_curr);
        % reg_lambda = mu * xCurresNorm^2;

        % reset timer
        timetic_riemna = tic();
        
        % the main update
        % transport
        for ii = 1 : m
            delta_xHistory{ii} = M.transp(xPre, xCur, delta_xHistory{ii});
        end
        delta_rHistory{curr_idx} = r_curr - M.transp(xPre, xCur, r_pre); 
        if iter >= 2
            for ii = 2 : m
                temp_idx = mod(curr_idx - 1 + ii - m, options.memory);
                if temp_idx == 0
                    temp_idx = options.memory;
                end
                delta_rHistory{temp_idx} = M.transp(xPre, xCur, delta_rHistory{temp_idx});
            end
        end
        
%         % compute tilde x
%         tildeX_memo = cell(1, m);
%         for ii = m : -1 : 1
%             temp_idx = mod(curr_idx - 1 + ii - m, options.memory);
%             if temp_idx == 0
%                 temp_idx = options.memory;
%             end
%             if ii == m
%                 tildeX_memo{temp_idx} = - delta_xHistory{temp_idx};
%             else
%                 pre_idx = mod(temp_idx + 1, options.memory);
%                 if pre_idx == 0
%                     pre_idx = options.memory;
%                 end
%                 tildeX_memo{temp_idx} = tildeX_memo{pre_idx}...
%                                         - delta_xHistory{temp_idx};
%             end
%         end

        % compute Rmat
        delta = c1 * M.norm(xCur, r_curr); % there is some issue here...
        Rmat = zeros(m);
        b = zeros(m, 1);
        for ii = 1 : m
            for jj = ii : m
                Rmat(ii,jj) = M.inner(xCur, delta_rHistory{ii}, delta_rHistory{jj}) ...
                              + delta * M.inner(xCur, delta_xHistory{ii}, delta_xHistory{jj});
                Rmat(jj,ii) = Rmat(ii,jj);
            end
            b(ii) = M.inner(xCur, r_curr, delta_rHistory{ii});%/ M.norm(xCur, curstep)^2;
        end
        % % normalization to ensure c is scale invariant
        % Rmat = Rmat / norm(Rmat); 

        gamma = computecoeff(Rmat, b, 0);
        % r_hat = curstep + do_average(delta_rHistory, gamma, m);
        bar_delta_x = beta * r_curr - do_average(delta_xHistory, gamma, m)...
                      - beta * do_average(delta_rHistory, gamma, m);
        

        % bar_delta_x = r_hat + do_average(tildeX_memo, gamma, m);
        tempx = M.retr(xCur, bar_delta_x);
        [tempcost, ~] = getCostGrad(problem, tempx);
        is_descent = (tempcost <= xCurCost);

        temp_idx = mod(curr_idx + 1, options.memory);
        if temp_idx == 0
             temp_idx = options.memory;
        end
        
        % verify if it's descent
        if is_descent
            %disp("yes")
            newx = tempx;
            % compute new grad
            [newcost, newgrad] = getCostGrad(problem, newx);
            newgradnorm = problem.M.norm(newx, newgrad);
            
            delta_xHistory{temp_idx} = bar_delta_x;
        else
            newx = problem.M.retr(xCur, curstep);
            % compute new grad
            [newcost, newgrad] = getCostGrad(problem, newx);
            newgradnorm = problem.M.norm(newx, newgrad);
            desc_dir = problem.M.lincomb(xCur, -1, xCurGradient);
            newstep = problem.M.lincomb(xCur, stepsize, desc_dir);
            
            delta_xHistory{temp_idx} = newstep;
        end
        
        % Save stats in a struct array info.
        stats = savestats();
        %keyboard;
        info(iter+1) = stats;
        info = info(1:iter+1);
        iter = iter + 1; 
    end



    %% helper
    function c = computecoeff(Rmat, b, lambda)
        sizeR = size(Rmat, 1);
%         reg_Rmat = Rmat + 2 * eye(sizeR) * lambda;
%         c = - reg_Rmat\(2 * b);

        c = lsqminnorm(Rmat +  eye(sizeR) * lambda,-b);

    end
    
    function res = do_average(lst, alpha, length)
        res = zeros('like', lst{1});
        for i = 1:length
            res = res + lst{i} * alpha(i);
        end
    end

    function stats = savestats()
        stats.iter = iter;
        stats.cost = xCurCost;
        stats.gradnorm = xCurGradNorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic_riemna);
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic_riemna);
        end
        stats = applyStatsfun(problem, xCur, [], [], options, stats);
    end

end
    

