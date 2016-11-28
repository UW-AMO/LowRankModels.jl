#====================================================================
  Using Trim-Glrm train movielens data
  - load data
  - partition data into training set and testing set
  - train data
  - test data
  - compare the result with the non-trimming version?
====================================================================#
using LowRankModels
#--------------------------------------------------------------------
# Load Data
#--------------------------------------------------------------------
println("load data...")
fid = open("./ratings.bin","r")
l   = read(fid,Int64,1)[1]          # total number of measurements
uid = read(fid,Int64,l)             # user ID
mid = read(fid,Int64,l)             # movie ID
rat = read(fid,Int64,l)             # corresponding ratings
close(fid)
#--------------------------------------------------------------------
# Partition Data
#--------------------------------------------------------------------
println("partition data...")
srand(123)
chaos  = randperm(l)
ind_tr = chaos[1:l>>2]
ind_te = chaos[l>>2+1:end]
# training set
uid_tr = view(uid,ind_tr)
mid_tr = view(mid,ind_tr)
rat_tr = view(rat,ind_tr)
R_tr   = sparse(uid_tr,mid_tr,rat_tr)   # training rating matrix
#--------------------------------------------------------------------
# Train Data
#--------------------------------------------------------------------
println("-----------------------train data------------------------")
m,n  = size(R_tr)
k    = 10                           # low-rank
h    = floor(Int64,m)           # trimming constant
@show(m,n,k,h)
loss = QuadLoss()
reg  = ZeroReg()                    # using zeroReg for experiment
glrm = GLRM(R_tr,loss,reg,reg,k,h)  # trim half the data
parm = SparseProxGradParams(1.0,100,1,1.0e-5,1.0e-3)
U, M, W, ch = fit!(glrm, parm)
#--------------------------------------------------------------------
# Test Data
#--------------------------------------------------------------------
println("------------------------test data------------------------")
# views for user and movie matrix
vu  = [view(U,:,i) for i = 1:m]
vm  = [view(M,:,j) for j = 1:n]
# for testing, we report MSE
mse = 0.0
for ind ∈ ind_te
    mse += (dot(vu[uid[ind]],vm[mid[ind]]) - rat[ind])^2
end
mse /= length(ind_te)
@printf("MSE: %1.5e\n",mse)
@show(sum(W))
#--------------------------------------------------------------------
# Plot W
#--------------------------------------------------------------------
# plot trimming vector
using PyPlot
plot(1:m,W,".b")
title("Trimming Weight Matrix")
savefig("trim_w.eps")
println("save trim_w.eps")