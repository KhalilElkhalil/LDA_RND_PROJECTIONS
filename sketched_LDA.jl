@everywhere using Distances
@everywhere using BenchmarkTools
@everywhere using MAT
#@everywhere using MTH229
@everywhere using SpecialFunctions
@everywhere include("/Users/macbookpro/Dropbox/Large_deviatinos/module_projections.jl")

# CDF of a standard Gaussian
@everywhere Phi(x) = 1/2*(1 + erf(x/sqrt(2)))

""" Data stats for artificial data """
p = 1000; # data dimension
#Sigma = eye(p); # identity model
rho = 0.4; # correlation factor
v = 0:p-1;
exponent = abs(v*ones(p)' - ones(p)v');
Sigma = (rho*ones(p, p)).^exponent; # Sigma_{i,j} = rho^(abs(i-j))
mu0 = zeros(p);
mu1 = mu0 + 3*ones(p) / sqrt(p);
mu = mu0 - mu1;
eigens, eigvect = eig(Sigma);
inv_Sigma = eigvect * diagm(ones(p)./eigens) * eigvect';
Sigma_r = eigvect * diagm(sqrt.(eigens)) * eigvect';

""" Experiments on Gaussian data """

n = 500;
n0 = Int64(ceil(n/2));
n1= n-n0;
X0 = mu0 * ones(1, n0) + Sigma_r*randn(p, n0);
X1 = mu1 * ones(1, n1) + Sigma_r*randn(p, n1);
X = [X0 X1];
y = [zeros(n0);ones(n1)]; # labels
y_lda = Fisher_LDA(X, mu0, mu1, inv_Sigma);
risk = sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_lda))) / n;

# random projections
d = 200;
avg = 0;
for it = 1:10
    W = randn(d, p);
    y_proj = Fisher_LDA(W*X, W*mu0, W*mu1, inv(W*Sigma*W'));
    risk_proj = sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_proj))) / n;
    avg = avg + risk_proj;
end
avg/10;

eigx, _ = eig(X*X'/n);
delx = fixed_point(eigx, p, d)

#=
 """ Experiments on MNIST data """
# choose the class labels by digit0 and digit1
digit0 = 8;
digit1 = 9;
train0, test0, train1, test1 = MNIST_data(digit0, digit1);
p, n0_train = size(train0);
_, n1_train = size(train1);
_, n0 = size(test0);
_, n1 = size(test1);
n = n0+n1;
mu0_ = 1/n0_train * train0*ones(n0_train); # sample mean
mu1_ = 1/n1_train * train1*ones(n1_train); # sample mean
mu_ = mu0_-mu1_;
# centering the data
train0 = train0 - mu0_*ones(n0_train)';
train1 = train1 - mu1_*ones(n1_train)';
train = [train0 train1];
# Pooled sample covariance matrix
Sigma_ = 1/(n0_train+n1_train-2)*(train*train');
eigens_, eigvect_ = eig(Sigma_);
eigens_ = eigens_ + 0.01*ones(p);
inv_Sigma_ = eigvect_ * diagm(ones(p)./eigens_) * eigvect_';
#inv_Sigma_ = inv(Sigma_ + 0.01*eye(p));
test = [test0 test1];
y = [zeros(n0);ones(n1)]; # the class labels
y_lda = Fisher_LDA(test, mu0_, mu1_, inv_Sigma_); # output of LDA
# evaluate the risk
risk = sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_lda))) / n;
risk_th = th_risk(p, p, eigens_, eigvect_, mu_);
# random projections
d = 100;
W = randn(d, p);
y_proj = Fisher_LDA(W*test, W*mu0_, W*mu1_, inv(W*Sigma_*W'));
risk = sum(xor(convert(Array{Int64}, y), convert(Array{Int64}, y_proj))) / n;
risk_th = th_risk(p, d, eigens_, eigvect_, mu_);
=#
