clear; clc;
close all;


fid = 1;
fprintf(fid,'==============================================================================================\n');

seed = round(rand() * 10000000);
rng(seed);


%% Rayleigh quotient minimization on the S^{n-1}
% min x'Ax s.t. x\in S^{n-1}
n = 10;
A = randn(n,n);
A = .5*(A + A');

%% 
f = @(x) fval(x,A);
grad = @(x) Rgrad(x,A);
R = @(x,eta) Retr(x,eta);

%R = @(x,eta) exponential(x,eta);

Tran = @(x1,v,d) Transp(x1,v,d);
IsoTran = @(x,v,d) isometricTransp(x,v,d);

x0 = randn(n,1); 
x0 = x0/norm(x0,'fro');


err = 1e-10; N = 5000; % Max iteration
%% Gradient descent
L = norm(A);
stepSize = 1/(5*L);
[x_gd, gd_time, gd_f, gd_grad] = Rie_gd(x0, f, grad,R, stepSize, N, err);

%% Rie_AA
m = 10;
c = 0.9;
stepSize = .05;

[x_RAA, RAA_f, RAA_gf, sindex, array_lambda] = Rie_AA_sphere(x0, f, grad, R, Tran, stepSize, err, N, m ,c);

% min f(x) = x^{T}A x s.t. x\in S^{n-1}
% The optimal solution is the eigenvector w.r.t the smallest eigenvalue
[V, D] = eig(A);
disp([V(1:10,1), x_gd(1:10), x_RAA(1:10)])
disp([f(V(:,1)), f(x_gd), f(x_RAA)])

figure(1);
plot(1:length(gd_f), gd_f,'r-','LineWidth',1.5);
hold on
plot(1:length(RAA_f), RAA_f,'bo--','LineWidth',1.5)
hold on
legend({'RGD','RAA'});
xlabel('Iteration');
ylabel('fval');

figure(2)
semilogy(1:length(gd_grad), gd_grad, 'r-','LineWidth',1.5);
hold on
semilogy(1:length(RAA_gf),RAA_gf,'bo--','LineWidth',1.5);
legend({'RGD','RAA'});
xlabel('Iteration');
ylabel('|grad|');

function [output] = fval(x, A)
   output = x'* A*x;
end

function [output] = Rgrad(x, A)
   Ax = A*x;
   output = 2*(Ax- (x'*Ax)*x);
end

function [output] = Retr(x, eta)
   x_eta = x + eta;
   output = x_eta/norm(x_eta,'fro');
end

% Exponential on the sphere
function y = exponential(x, d)

%     if nargin == 2
%         % t = 1
%         td = d;
%     else
%         td = t*d;
%     end
    td = d;
    nrm_td = norm(td, 'fro');
    
    % Former versions of Manopt avoided the computation of sin(a)/a for
    % small a, but further investigations suggest this computation is
    % well-behaved numerically.
    if nrm_td > 0
        y = x*cos(nrm_td) + td*(sin(nrm_td)/nrm_td);
    else
        y = x;
    end

end
   


function [output] = Transp(x1,v,eta)
   x2 = Retr(x1, v);
   output = eta - x2*(x2'*eta);
end 

function [output] = isometricTransp(x1, v, d)
   % Isometric vector transport of d from the tangent space at x1 to x2 = Retr(x1, v).
   % This is actually a parallel vector transport
   %x2 = Retr(x1,v);
   %v = logarithm(x1, x2);
   dist_x1x2 = norm(v, 'fro');
   if dist_x1x2 > 0
       u = v / dist_x1x2;
       utd = u(:)'*d(:);
       output = d + (cos(dist_x1x2)-1)*utd*u ...
           -  sin(dist_x1x2)  *utd*x1;
   else
       % x1 == x2, so the transport is identity
       output = d;
   end

end

% function [output] = Transp(x1,x2,eta)
%    % x2 = Retr(x1, xi)
%    output = eta - x2*(x2'*eta);
% end 

% function Td = isometricTransp(x1, x2, d)
%         v = logarithm(x1, x2);
%         dist_x1x2 = norm(v, 'fro');
%         if dist_x1x2 > 0
%             u = v / dist_x1x2;
%             utd = u(:)'*d(:);
%             Td = d + (cos(dist_x1x2)-1)*utd*u ...
%                     -  sin(dist_x1x2)  *utd*x1;
%         else
%             % x1 == x2, so the transport is identity
%             Td = d;
%         end
%     end
% 
% function v = logarithm(x1, x2)
%         d = x2 - x1;
%         v = d - x1*(x1'*d);
%         %v = M.proj(x1, x2 - x1);
%         di = dist(x1, x2);
%         % If the two points are "far apart", correct the norm.
%         if di > 1e-6
%             nv = norm(v, 'fro');
%             v = v * (di / nv);
%         end
%     end
% 
% function d = dist(x, y)
%         
%         % The following code is mathematically equivalent to the
%         % computation d = acos(x(:)'*y(:)) but is much more accurate when
%         % x and y are close.
%         
%         chordal_distance = norm(x - y, 'fro');
%         d = real(2*asin(.5*chordal_distance));
%         
%         % Note: for x and y almost antipodal, the accuracy is good but not
%         % as good as possible. One way to improve it is by using the
%         % following branching:
%         % % if chordal_distance > 1.9
%         % %     d = pi - dist(x, -y);
%         % % end
%         % It is rarely necessary to compute the distance between
%         % almost-antipodal points with full accuracy in Manopt, hence we
%         % favor a simpler code.
%         
%     end











