function [x,time_t, error_history,error_g] = rna(x0,funval,gradient, stepsize,inner_iter, outer_iter)
% Initialization
% Set x = x0
x = x0;
% Optimal step size for convex functions
stepSize =  stepsize;
k=inner_iter;
N=outer_iter;
error_history = zeros(k*N,1);
time_t = zeros(k*N,1);
dim=length(x0);
x_history = zeros(dim,k);
iter = 0;
total_time=0;

for i=1:N
    
    % perform k gradient steps
    for j=1:k
        tic;
        % gradient step
        grad=gradient(x);
        x = x-stepSize*grad; 
        
        % Store iterate x for extrapolation
        x_history(:,j) = x;
                
        % Record norm(x-xstar)
        iter = iter + 1;
        t2=toc;
        total_time=total_time+t2;
        error_history(iter) = funval(x);
        error_g(iter)=norm(stepSize*grad);
        time_t(iter)=total_time;
    end
    % Restart using x0 = x_extrapolated
       tic;
    warning off % avoid ('Matrix is close to singular or badly scaled')
    x = adaptive_rna(funval,x_history);
    warning on
     t2=toc;
    total_time=total_time+t2;
end
end
 
