clear all

%Graph  and coeffs loading

coeff = table2array(readtable('coeff.csv'));
n = size(coeff,1);

ground_truth = table2array(readtable('clusters.csv'));
%ground_truth = ground_truth;
A =  table2array(readtable('graph.csv'));
A = .5*(A+A'); % To avoid numeric precision problems


% 1 vs all useful data structures
classes = 1:3;
results = zeros(n,length(classes),2);

for class = classes

%Labeled sample
cluster_class= find(ground_truth == class);
cluster_not_class = find(ground_truth ~= class);

l_class = datasample(cluster_class,35,'Replace',false);
l_not_class = datasample(cluster_not_class,35,'Replace',false);
l = sort(cat(1,l_class,l_not_class));
sample_size = length(l_class) + length(l_not_class);

%Dissimilarity introduction
diss_number = 6;
for diss = 1:diss_number
    i = datasample(l_class,1);
    j = datasample(l_not_class,1);
    G1 = coeff(i);
    G2 = coeff(j);
    A(i,j) = -sqrt(sum((G1 - G2) .^ 2));
    A(j,i) = A(i,j);
end

%Signed Degree
D = diag(abs(A)*ones(n,1));

%Signed Laplacian
L = D-A;



%Building the constraint matrix
Aeq = zeros(sample_size,n);
beq = zeros(n,1);
row_counter = 1;
for el = 1:length(l)
   Aeq(row_counter,l(el))=1; 
   if ismember(l(el),l_not_class)
       beq(l(el)) = -1;
   else
       beq(l(el)) = 1;
   end
   row_counter = row_counter +1;
end
beq = nonzeros(beq);

%Hyperparameters initializations
lambda_minus = 0;
lambda_plus = 0;

%N(l_minus) and N(l_plus) computing
N_l_minus = [];
N_l_plus = [];
V_diff_l = setdiff(1:n,l);
for i = l_not_class
    for j = V_diff_l
        if A(i,j) > 0
            N_l_minus = [N_l_minus j];
        end
    end
end
N_l_minus = unique(N_l_minus);

for i = l_class
    for j = V_diff_l
        if A(i,j) > 0
            N_l_plus = [N_l_plus j];
        end
    end
end
N_l_plus = unique(N_l_plus);

% N_minus and N_plus computing

N_minus = setdiff(N_l_minus,N_l_plus);
N_plus = setdiff(N_l_plus,N_l_minus);


%ADMM Algorithm
end_flag = 1;
x_min = .8;

% Optimization parameters
max_iter = 100;
iter = 1;
start = -1 + (1+1)*rand(n,1);

start(l_not_class) = -1;
start(l_class) = 1;

while(end_flag == 1)

%Induced Laplacian Form Penalized Minimization
objFun = @(x) x'*L*x + lambda_minus*sum(abs(ones(length(N_minus),1)+x(N_minus))) + lambda_plus*sum(abs(ones(length(N_plus),1)-x(N_plus))) ;
options=optimoptions('fmincon','MaxFunctionEvaluations',2e10,'OptimalityTolerance',1e-2,'ConstraintTolerance',1e-1,'StepTolerance',1e-14,'MaxIteration',4e4); % Optimization Options
x_star =fmincon(objFun,start,[],[],Aeq,beq,[],[],[],options);
start = x_star;

M_minus = find(x_star(N_minus)<0);
M_plus = find(x_star(N_plus)>0);
x_minus = min(abs(x_star(M_minus)));
x_plus = min(abs(x_star(M_plus)));

end_flag = 0;

if isempty(M_minus) || x_minus < x_min
    lambda_minus = lambda_minus + 1;
    end_flag = 1;
end

if isempty(M_plus) || x_plus < x_min
    lambda_plus = lambda_plus + 1;
    end_flag = 1;
end

if iter > max_iter
    end_flag = 0;
end

iter = iter +1;
end

% Cluster definition
results(:,class,2) = x_star;

x_star(x_star < 0)=0;
x_star(x_star > 0)=class;

results(:,class,1) = x_star;

end

% 1vsALL scoring to solve ambiguities
final_clustering = zeros(n,1);
for row = 1:n
    if nnz(results(row,:,1)) > 1 ||  nnz(results(row,:,1)) == 0
        [~,class] = find(results(row,:,2) == max(results(row,:,2)));
        final_clustering(row) = class;
    else
        final_clustering(row) = find(results(row,:,1));
    end

end


%Accuracy computing
errs = 0;
good = 0;
for i = 1:length(ground_truth)
if final_clustering(i) == ground_truth(i)
    good = good +1;
else
errs = errs +1;
end
end


accuracy = good/(errs+good)

clustering_result_final_df = cat(2,coeff,final_clustering);

% Saving the trained model for classification
save('clustering_result','clustering_result_final_df');