function Xbar = mfd_average(M, X, c)
% implement spd averging on X with weight c
% Use weighted average recursion where retr and ivnerse retraction
% are defined in manopt
%
% To use exponential map, inverse exponential map, please redefine the
% factory file by setting M.retr = @exponential, M.invretr = @logarithm
% where exponential, logarithm are well-defined functions for exponential 
% map and inverse exponential map
%
% Input:
%     M : a manifold struct
%     X : a cell like array of size n
%     c : a vector of size nx1 corresponding to the order of X

% Original author: Andi Han

    n = length(X);
    n_ = length(c);
    assert(n == n_);

    X1 = X{1};
    
    Xtilde = X1;
    gammas = c ./ cumsum(c);
    for i = 2 : n
        temp = M.invretr(Xtilde, X{i});
        step = M.lincomb(Xtilde, gammas(i), temp);
        Xtilde = M.retr(Xtilde, step);
    end
    Xbar = Xtilde;


end

