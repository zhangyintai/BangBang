format long
digits(128)
% rng(8)
Sx = [[0, 1]; [1, 0]];
Sy = [[0, -1j]; [1j, 0]];
Sz = [[1, 0]; [0, -1]];
Id = eye(2);
    
hc = 3.04438;
hx = 3.1;
hz = 0;
H1 = -hx * Sx - hz * Sz;    
H2 = - kron(Sz, Sz);
H_ising = (kron(H1, Id) + kron(Id, H1)) / 4.0 + H2;


ntu_tol = 1e-9;
pinv_tol = 1e-10;
svd_tol = 1e-12;
ctm_svd_tol = 5e-4;
% ctm_svd_tol = -1;
Etol = 1e-5;
d = 2;
D = 6;
chi = 24;

test = PEPS_test_5;
test.SetValues(1 / sqrt(2), 1 / sqrt(2), 1 / sqrt(2), 1 / sqrt(2), d, D, chi);
tlist = (0:0.1:2);
Sx_list = [];
    
% for t = tlist
%     test_temp = copy(test);
%     [U_left, U_right] = test_temp.HamiltonianExp(H_ising, t / 10, 1);
%     for ii = (0:10)
%         for bond_dir = (0:3)
%             test_temp.ApplyGate(U_left, U_right, bond_dir, "NTU", ntu_tol, pinv_tol);
%         end5
%     end
%     [energy, rho] = test_temp.Energy(H_ising, etol, pinv_tol);
%     sx = trace(rho * (kron(Sx, Id) + kron(Id, Sx))) / 2;
%     disp(sx)
%     Sx_list = [Sx_list, real(sx)];
%     delete(test_temp)zz 
% end
P = 6;
tau = 0.01;
xdim = 2 * P;
epsilon = 1e-5;
total_time = 0.7;

alpha0 = [ones(1,P - 1) * tau, tau / 2];
beta0 = (sin((linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)-0.5) * pi) + 1) * 0.5 * tau;

% x0 = [ones(1,P - 1) * tau, tau / 2, (sin((linspace(1/(2 * P), (2 * P - 1)/(2 * P), P)-0.5) * pi) + 1) * 0.5 * tau];
% x0 = [0.794531, 0.747656, 0.5875, 0.2625, 0.202724, 0.279713, 0.307787, 0.322276];
tau_ub = inf;

% P = 8
% x0 = [0.559375000000000, 0.231250000000000, 0.149218750000000, ...
%     0.262500000000000,   0.387500000000000,   0.325000000000000, ...
%     0.418750000000000,   0.256250000000000, 0.064421471959677, ...
%     0.110603038769745, 0.044442976698040, 0.080490967798387, ...
%     0.119509032201613, 0.155557023301960, 0.183146961230255, 0.229328528040323];
% [energy, rho] = test.BangBang_Ising(H1, 1, ntu_tol, pinv_tol, etol, P, 1, x0);

x0 = [];
for i = 1:P
    x0 = [x0, alpha0(i), beta0(i)];
end

x0 = [0.197904000000000   0.059100200000000   0.134115000000000   0.096849500000000   0.160902000000000   0.114636000000000   0.077359900000000   0.135119000000000  0.008034650000000   0.009067040000000 0 0];

% P = 6
% x0 = [0.55781250, 0.31171875, 0.36250000, ...
%     0.53437500, 0.50312500, 0.22812500, ...
%     0.09104863, 0.12987148, 0.11117714, ...
%     0.18882286, 0.27169102, 0.24801387];
% objective = @(x)test.BangBang_Ising_(H1, 1, ntu_tol, svd_tol, pinv_tol, ctmtol, P, 0, x);
constr = @(x)Constraint(x, total_time);
% objective = @(x)test.BangBang_Ising_(H1, 1, ntu_tol, svd_tol, pinv_tol, Etol, ctm_svd_tol, P, 0, round([x(1:2:2*P), x(2:2:2*P)], 6), 1);
objective = @(x)test.BangBang_Ising_(H1, 1, ntu_tol, svd_tol, 1e-9, Etol, ctm_svd_tol, P, 0, [x(1:2:2*P), x(2:2:2*P)], 1, 0);
objective_test = @(x)test.BangBang_Ising_(H1, 1, ntu_tol, svd_tol, 1e-9, Etol, ctm_svd_tol, P, 1, [x(1:2:2*P), x(2:2:2*P)], 2, 0);
objective_tangentmps = @(x)test.BangBang_Ising_(H1, 1, ntu_tol, svd_tol, 1e-9, Etol, ctm_svd_tol, P, 0, [x(1:2:2*P), x(2:2:2*P)], 2, 0);
pyenv("ExecutionMode","OutOfProcess");
objective_yastn = @(x)(pyrunfile("BangBang_yastn_matlab_.py", "test", D=D, chi = chi, ...    
    a=1 / sqrt(2), b=1 / sqrt(2), svd_tol = svd_tol, ctm_tol = ctm_svd_tol, HX = hx, HZ = hz, P = P, ...
        imagine_real = 1, max_sweeps = 1000, x0 = [x(1:2:2*P), x(2:2:2*P)], Etol = Etol) / 2);
ub = repmat([min(tau_ub, pi / hx), tau_ub], 1, P);
ub = ones(1, 2 * P) * 0.2;

lb = ones(1, 2 * P) * -0.2;

% options = optimoptions('ga','Display', 'iter', 'PlotFcn', @gaplotbestf);
% [x,fval,exitflag,output] = ga(objective, xdim, [], [], [], [], lb, ub, [], options);

% options = optimoptions("fmincon", "PlotFcn", {@optimplotx, @optimplotfval, @optimplotstepsize});
% [x,fval,exitflag,output] = fmincon(objective, x0, [], [], [], [], lb, ub, [], options)

% options = optimoptions('patternsearch', 'Algorithm', 'nups-gps', 'Display','iter','PlotInterval', 1, 'PlotFcn',{@psplotfuncount, @psplotbestf,@psplotbestx});
% [x,fval,exitflag,output] = patternsearch(objective, x0, [], [], [], [], lb, ub, options);

% options = optimoptions('simulannealbnd','Display','iter','PlotInterval', 1, 'MaxIterations', 1000, 'PlotFcn',{@saplotbestx, @saplotbestf, @saplotx, @saplotf});
% [x,fval,exitflag,output] = simulannealbnd(objective, x0, lb, ub, options);

% options = optimoptions('particleswarm', 'Display', 'iter', 'PlotFcn', @pswplotbestf);
% [x,fval,exitflag,output]=particleswarm(objective, 2 * P, lb, ub, options);


% [x,fval,exitflag,output] = surrogateopt(objective, lb, ub);
% options = optimoptions('surrogateopt','Display','iter','PlotFcn');
