function [x, cost, info, options] = RAGD(problem, x, options)
% Riemannian accelerated gradient descent in "An Estimate Sequence for 
% Geodesically Convex Optimization". Refer to the code in https://github.com/jungbinkim1/RNAG

% The algorithm requires retr and invretr, to be properly defined. To use
% exponential map. Please redefine a factory file with the field
% retr/invretr defined as exp/log.

    % Verify that the problem description is sufficient for the solver.
    if ~canGetCost(problem)
        warning('manopt:getCost', ...
                'No cost provided. The algorithm will likely abort.');
    end
    if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
        % Note: we do not give a warning if an approximate gradient is
        % explicitly given in the problem description, as in that case the
        % user seems to be aware of the issue.
        warning('manopt:getGradient:approx', ...
               ['No gradient provided. Using an FD approximation instead (slow).\n' ...
                'It may be necessary to increase options.tolgradnorm.\n' ...
                'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
        problem.approxgrad = approxgradientFD(problem);
    end

    % Set local defaults here.
    localdefaults.minstepsize = 1e-10;
    localdefaults.maxiter = 1000;
    localdefaults.tolgradnorm = 1e-6; 

    localdefaults.stepsize = 0.1; % should be set as 1/L
    localdefaults.mu = 1;
    
    % Merge global and local defaults, then merge w/ user options, if any.
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    timetic = tic();
    
    % If no initial point x is given by the user, generate one at random.
    if ~exist('x', 'var') || isempty(x)
        x = problem.M.rand();
    end
    
    % Create a store database and get a key for the current x.
    storedb = StoreDB(options.storedepth);
    key = storedb.getNewKey();
    
    % Compute objective-related quantities for x.
    [cost, grad] = getCostGrad(problem, x, storedb, key);
    gradnorm = problem.M.norm(x, grad);


    % Iteration counter.
    % At any point, iter is the number of fully executed iterations so far.
    iter = 0;
    
    % Save stats in a struct array info, and preallocate.
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];

    if options.verbosity >= 2
        fprintf(' iter\t               cost val\t    grad. norm\n');
    end

    M = problem.M;
    
    % parameters
    stepsize = options.stepsize;
    mu = options.mu;
    beta = sqrt(mu*stepsize)/5;
    alpha = ( sqrt(beta^2 + 4*(1+beta)*mu*stepsize) - beta)/2;
    gamma = mu * ( sqrt(beta^2 + 4*(1+beta)*mu*stepsize) - beta )/ ...
        (sqrt(beta^2 + 4*(1+beta)*mu*stepsize) + beta);
    gamma_bar = (1+beta)*gamma;
    v = x;
    
    % Start iterating until stopping criterion triggers.
    while true

        % Display iteration information.
        if options.verbosity >= 2
            fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
        end
        
        % Start timing this iteration.
        timetic = tic();
        
        % Run standard stopping criterion checks.
        [stop, reason] = stoppingcriterion(problem, x, options, ...
                                                             info, iter+1);

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
        
        temp = M.invretr(x, v);
        tempcoeff = alpha*gamma/(gamma+alpha*mu);
        y = M.retr(x, M.lincomb(x, tempcoeff, temp));
        
        [~, grady] = getCostGrad(problem, y);
        dir = M.lincomb(y, -stepsize, grady);
        x = M.retr(y, dir);
        
        tempfinal = M.invretr(y, v);
        coeff1 = (1-alpha)*gamma/gamma_bar;
        coeff2 = -alpha/gamma_bar;
        
        v = M.retr(y, M.lincomb(y, coeff1, tempfinal, coeff2, grady));
        
        
        [cost, grad] = getCostGrad(problem, x);
        gradnorm = problem.M.norm(x, grad);

        % iter is the number of iterations we have accomplished.
        iter = iter + 1;
        
        % Log statistics for freshly executed iteration.
        stats = savestats();
        info(iter+1) = stats;
    end
    
    info = info(1:iter+1);
    
    
    % Routine in charge of collecting the current iteration stats.
    function stats = savestats()
        stats.iter = iter;
        stats.cost = cost;
        stats.gradnorm = gradnorm;
        if iter == 0
            stats.stepsize = NaN;
            stats.time = toc(timetic);
        else
            stats.stepsize = stepsize;
            stats.time = info(iter).time + toc(timetic);
        end
        stats = applyStatsfun(problem, x, [], [], options, stats);
    end
    
    
end

