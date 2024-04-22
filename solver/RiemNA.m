function [xCur, xCurCost, info, options] = RiemNA(problem, x0, options)
% Implement the offline version of RiemNA using the iterates from steepest
% descent. Notice that this implementation is built on the iterates from
% steepest descent and should be a standalone solver. 
%
% function [x, cost, info, options] = RiemNA(problem)
% function [x, cost, info, options] = RiemNA(problem, x0)
% function [x, cost, info, options] = RiemNA(problem, x0, options)
% function [x, cost, info, options] = RiemNA(problem, [], options)


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
    averagefn = options.average;
    

    
    % To make sure memory in range [2, Inf)
    % how many residual stored, it needs to be larger than 1, otherwise it
    % will be the same as RGD.
    options.memory = max(options.memory, 2); 
    if options.memory == Inf
        if isinf(options.maxiter)
            options.memory = 10000;
            warning('RiemNA:memory', ['options.memory and options.maxiter' ...
              ' are both Inf; options.memory has been changed to 10000.']);
        else
            options.memory = options.maxiter;
        end
    end
    
    M = problem.M;
        
    % __________Initialization of variables______________
    % Create a random starting point if no starting point is provided.
    if ~exist('x0', 'var')|| isempty(x0)
        newx = M.rand();
    else
        newx = x0;
    end

    % Query the cost function and its gradient
    [newcost, newgrad] = getCostGrad(problem, newx);
    newgradnorm = problem.M.norm(newx, newgrad);

    % Iteration counter.
    iter = 0;

    % store the residual and function value and R matrix
    rHistory = cell(1, options.memory); 
    xHistory = cell(1, options.memory);
    Rmat = zeros(options.memory, options.memory);

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
    
    % start timing
    timetic_riemna = tic();


    % Start iterating until stopping criterion triggers.
    while true
        
        % loop over gd
        for ii = 1 : options.memory

            % update current pt
            xCur = newx;
            xCurGradient = newgrad;
            xCurGradNorm = newgradnorm;
            xCurCost = newcost;

            % Display iteration information.
            if options.verbosity >= 2
                fprintf('%5d\t%+.16e\t%.8e\n', iter, xCurCost, xCurGradNorm);
            end
            
            % Save stats in a struct array info.
            stats = savestats();
            %keyboard;
            info(iter+1) = stats;
            % reset timer
            timetic_riemna = tic(); 


            %% stopping criteria
            % Run standard stopping criterion checks.
            [stop, reason] = stoppingcriterion(problem, xCur, options, ...
                                                                 info, iter+1);
    
            % If none triggered, run specific stopping criterion check.
            if ~stop && stats.stepsize < options.minstepsize
                stop = true;
                reason = sprintf(['Last stepsize smaller than minimum '  ...
                                  'allowed; options.minstepsize = %g.'], ...
                                  options.minstepsize);
            end
            if stop
                break;
            end


            %% gd
            % Pick the descent direction as minus the gradient.
            desc_dir = problem.M.lincomb(xCur, -1, xCurGradient);
            stepsize = options.stepsize;
            curstep = problem.M.lincomb(xCur, stepsize, desc_dir);
            newx = problem.M.retr(xCur, curstep);
            
            % Compute the new cost-related quantities for x
            [newcost, newgrad] = getCostGrad(problem, newx);
            newgradnorm = problem.M.norm(newx, newgrad);
            iter = iter + 1;
        
            % update memory
            rHistory{ii} = curstep;
            xHistory{ii} = xCur;
        end
        if stop
            if options.verbosity >= 1
                fprintf([reason '\n']);
            end
            break;
        end


        % transport 
        for ii = 1 : options.memory - 1
            rHistory{ii} = M.transp(xHistory{ii}, xCur, rHistory{ii});
        end

        % compute Rmat
        for ii = 1 : options.memory
            for jj = ii : options.memory
                Rmat(ii,jj) = M.inner(xCur, rHistory{ii}, rHistory{jj});
                Rmat(jj,ii) = Rmat(ii,jj);
            end
        end
        % normalization to ensure c is scale invariant
        Rmat = Rmat / norm(Rmat); 

        c = computecoeff(Rmat, options.reg_lambda);
        newx = averagefn(problem.M, xHistory, c);
        % compute new grad
        [newcost, newgrad] = getCostGrad(problem, newx);
        newgradnorm = problem.M.norm(newx, newgrad);

        
    end
    info = info(1:iter+1);




    %% helper
    function c = computecoeff(Rmat, lambda)
        sizeR = size(Rmat,1);
        reg_Rmat = Rmat + eye(sizeR) * lambda;
        c = reg_Rmat\ones(sizeR,1);
        c = c/sum(c);
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

