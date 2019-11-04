% Load the clusters
load('clustering_result');

% Load the test and set up labels to evaluate accuracy
test = table2array(readtable('test.csv'));
not_labeled_cluster = clustering_result_final_df(:,1:(size(clustering_result_final_df,2)-1));
labels = clustering_result_final_df(:,size(clustering_result_final_df,2));
test_labels = test(:,size(test,2));
not_labeled_test = test(:,1:(size(test,2)-1));

% Majority voting using same criterion used to build the initial graph
for k = 1:139
spat_dist = knnsearch(not_labeled_cluster,not_labeled_test,'K',k,'Distance','minkowski','P',2);
prediction = zeros(size(test,1),1);
for row = 1:size(spat_dist,1)
   prediction(row) = mode(labels(spat_dist(row,:)));
end

%Accuracy computing
errs = 0;
good = 0;
for i = 1:length(test_labels)
if prediction(i) == (test_labels(i))
    good = good +1;
else
errs = errs +1;
end
end

accuracy(k) = good/(errs+good);
end
plot(accuracy)
[~,k] = find(accuracy == max(accuracy));