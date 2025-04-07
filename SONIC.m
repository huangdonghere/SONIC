function [preY] = SONIC(X,c,m,alpha,opts)
k=5;
if (~exist('opts','var'))
   opts = [];
end
if k>=m
    k = m-1;
end
Distance = 'sqEuclidean';  %(the default)
if isfield(opts,'Distance')
    Distance = opts.Distance;
end

view= size(X,2);
n = size(X{1},1);
NITER = 15;
IterMaxS = 50;

%% =====================   preparation =====================
%Maximum and minimum normalization
XX = [];
for i = 1:view
    X{i} = ( X{i}-repmat(min(X{i}),n,1) ) ./repmat(max(X{i})-min(X{i}),n,1);
    X{i}( isnan(X{i}) ) = 1;
    XX = [XX X{i}];
    d(i) = size(X{i},2); %The number of features in the i-th view
end

%initialization anchors
if n>10000
    rand('twister',5489);
    tmpIdx = randsample(n,10000,false);
    subfea = XX(tmpIdx,:);
else
    subfea = XX;
end

rand('twister',5489);
[~, AA] = litekmeans(subfea,m,'MaxIter', 20,'Replicates',2,'Distance',Distance); 

%Split
temp = 0;
for ia = 1:view
    A{ia} = AA(:, 1 + temp : d(ia) + temp );
    temp = temp + d(ia);
end

%initialize weighted_distX
SUM = zeros(n,m);
for i = 1:view
    distX_initial(:,:,i) =  full(L2_distance_1( X{i}',A{i}' ) );                  
    SUM = SUM + distX_initial(:,:,i);
end
for i = 1:view
    Wv(i) = 1/view;
end
distX = 1/view*SUM; 
[distXs, idx] = sort(distX,2);

%KNN initialize S
S = zeros(n,m);
rr = zeros(n,1);
for i = 1:n
    di = distXs(i,1:k+1);
    id = idx(i,1:k+1);
    S(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);               
end
[~,~,S,~,~,~] = coclustering_bipartite_fast1(S, c, IterMaxS,distX,alpha);
res = zeros(NITER + 1,1); 
res(1,1) = obj_value(X,A,S,view,alpha,Wv);
WvArray = zeros(view,NITER + 1);
WvArray(:,1) = Wv;

%% =====================  updating =====================

for iter = 1:NITER
     %update A
    for ia = 1:view
        for j = 1:m
            A{view}(j,:) = sum(S(:,j).*X{view})./sum(S(:,j));
        end
    end

    % update weighted_distX
    SUM = zeros(n,m);
    for i = 1:view
        distX_updated(:,:,i) =  full(L2_distance_1( X{i}',A{i}' ) );
        Wv(i) = 0.5/sqrt(sum(sum( distX_updated(:,:,i).*S+eps)));
        distX_updated(:,:,i) = Wv(i)*distX_updated(:,:,i) ;
        SUM = SUM + distX_updated(:,:,i);
    end
    distX = SUM;
    WvArray(:,iter + 1) = Wv;
    
    %update S
    [preY,~,S,~,~,~] = coclustering_bipartite_fast1(S, c, IterMaxS,distX,alpha);
    
    %Calculate the objective function value
    res(iter+1,1) = obj_value(X,A,S,view,alpha,Wv);
    if abs(res(iter+1,1) - res(iter,1)) < 1e-5 || norm(WvArray(:,iter + 1) - WvArray(:,iter),2) < 1e-5
        break
    end   
end

end
%obj_value function
function res = obj_value(X,A,S,view,alpha,Wv)
res = 0;
for t = 1:view
    TMP = Wv(t)*pdist2(X{t},A{t}).^2.*S;
    res = res + sum(sum(TMP));
end
res = res + alpha * norm(S,'fro')^2;
end

function d = L2_distance_1(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b



if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);
end

function [label, center, bCon, sumD, D] = litekmeans(X, k, varargin)
%LITEKMEANS K-means clustering, accelerated by matlab matrix operations.
%
%   label = LITEKMEANS(X, K) partitions the points in the N-by-P data matrix
%   X into K clusters.  This partition minimizes the sum, over all
%   clusters, of the within-cluster sums of point-to-cluster-centroid
%   distances.  Rows of X correspond to points, columns correspond to
%   variables.  KMEANS returns an N-by-1 vector label containing the
%   cluster indices of each point.
%
%   [label, center] = LITEKMEANS(X, K) returns the K cluster centroid
%   locations in the K-by-P matrix center.
%
%   [label, center, bCon] = LITEKMEANS(X, K) returns the bool value bCon to
%   indicate whether the iteration is converged.  
%
%   [label, center, bCon, SUMD] = LITEKMEANS(X, K) returns the
%   within-cluster sums of point-to-centroid distances in the 1-by-K vector
%   sumD.    
%
%   [label, center, bCon, SUMD, D] = LITEKMEANS(X, K) returns
%   distances from each point to every centroid in the N-by-K matrix D. 
%
%   [ ... ] = LITEKMEANS(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies
%   optional parameter name/value pairs to control the iterative algorithm
%   used by KMEANS.  Parameters are:
%
%   'Distance' - Distance measure, in P-dimensional space, that KMEANS
%      should minimize with respect to.  Choices are:
%            {'sqEuclidean'} - Squared Euclidean distance (the default)
%             'cosine'       - One minus the cosine of the included angle
%                              between points (treated as vectors). Each
%                              row of X SHOULD be normalized to unit. If
%                              the intial center matrix is provided, it
%                              SHOULD also be normalized.
%
%   'Start' - Method used to choose initial cluster centroid positions,
%      sometimes known as "seeds".  Choices are:
%         {'sample'}  - Select K observations from X at random (the default)
%          'cluster' - Perform preliminary clustering phase on random 10%
%                      subsample of X.  This preliminary phase is itself
%                      initialized using 'sample'. An additional parameter
%                      clusterMaxIter can be used to control the maximum
%                      number of iterations in each preliminary clustering
%                      problem.
%           matrix   - A K-by-P matrix of starting locations; or a K-by-1
%                      indicate vector indicating which K points in X
%                      should be used as the initial center.  In this case,
%                      you can pass in [] for K, and KMEANS infers K from
%                      the first dimension of the matrix.
%
%   'MaxIter'    - Maximum number of iterations allowed.  Default is 100.
%
%   'Replicates' - Number of times to repeat the clustering, each with a
%                  new set of initial centroids. Default is 1. If the
%                  initial centroids are provided, the replicate will be
%                  automatically set to be 1.
%
% 'clusterMaxIter' - Only useful when 'Start' is 'cluster'. Maximum number
%                    of iterations of the preliminary clustering phase.
%                    Default is 10.  
%
%
%    Examples:
%
%       fea = rand(500,10);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50);
%
%       fea = rand(500,10);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Replicates', 10);
%
%       fea = rand(500,10);
%       [label, center, bCon, sumD, D] = litekmeans(fea, 5, 'MaxIter', 50);
%       TSD = sum(sumD);
%
%       fea = rand(500,10);
%       initcenter = rand(5,10);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Start', initcenter);
%
%       fea = rand(500,10);
%       idx=randperm(500);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Start', idx(1:5));
%
%
%   See also KMEANS
%
%    [Cite] Deng Cai, "Litekmeans: the fastest matlab implementation of
%           kmeans," Available at:
%           http://www.zjucadcg.cn/dengcai/Data/Clustering.html, 2011. 
%
%   version 2.0 --December/2011
%   version 1.0 --November/2011
%
%   Written by Deng Cai (dengcai AT gmail.com)


if nargin < 2
    error('litekmeans:TooFewInputs','At least two input arguments required.');
end

[n, p] = size(X);


pnames = {   'distance' 'start'   'maxiter'  'replicates' 'onlinephase' 'clustermaxiter'};
dflts =  {'sqeuclidean' 'sample'       []        []        'off'              []        };
[eid,errmsg,distance,start,maxit,reps,online,clustermaxit] = getargs(pnames, dflts, varargin{:});
if ~isempty(eid)
    error(sprintf('litekmeans:%s',eid),errmsg);
end

if ischar(distance)
    distNames = {'sqeuclidean','cosine'};
    j = strcmpi(distance, distNames);
    j = find(j);
    if length(j) > 1
        error('litekmeans:AmbiguousDistance', ...
            'Ambiguous ''Distance'' parameter value:  %s.', distance);
    elseif isempty(j)
        error('litekmeans:UnknownDistance', ...
            'Unknown ''Distance'' parameter value:  %s.', distance);
    end
    distance = distNames{j};
else
    error('litekmeans:InvalidDistance', ...
        'The ''Distance'' parameter value must be a string.');
end


center = [];
if ischar(start)
    startNames = {'sample','cluster'};
    j = find(strncmpi(start,startNames,length(start)));
    if length(j) > 1
        error(message('litekmeans:AmbiguousStart', start));
    elseif isempty(j)
        error(message('litekmeans:UnknownStart', start));
    elseif isempty(k)
        error('litekmeans:MissingK', ...
            'You must specify the number of clusters, K.');
    end
    if j == 2
        if floor(.1*n) < 5*k
            j = 1;
        end
    end
    start = startNames{j};
elseif isnumeric(start)
    if size(start,2) == p
        center = start;
    elseif (size(start,2) == 1 || size(start,1) == 1)
        center = X(start,:);
    else
        error('litekmeans:MisshapedStart', ...
            'The ''Start'' matrix must have the same number of columns as X.');
    end
    if isempty(k)
        k = size(center,1);
    elseif (k ~= size(center,1))
        error('litekmeans:MisshapedStart', ...
            'The ''Start'' matrix must have K rows.');
    end
    start = 'numeric';
else
    error('litekmeans:InvalidStart', ...
        'The ''Start'' parameter value must be a string or a numeric matrix or array.');
end

% The maximum iteration number is default 100
if isempty(maxit)
    maxit = 100;
end

% The maximum iteration number for preliminary clustering phase on random
% 10% subsamples is default 10 
if isempty(clustermaxit)
    clustermaxit = 10;
end


% Assume one replicate
if isempty(reps) || ~isempty(center)
    reps = 1;
end

if ~(isscalar(k) && isnumeric(k) && isreal(k) && k > 0 && (round(k)==k))
    error('litekmeans:InvalidK', ...
        'X must be a positive integer value.');
elseif n < k
    error('litekmeans:TooManyClusters', ...
        'X must have more rows than the number of clusters.');
end


bestlabel = [];
sumD = zeros(1,k);
bCon = false;

for t=1:reps
    switch start
        case 'sample'
            center = X(randsample(n,k),:);
        case 'cluster'
            Xsubset = X(randsample(n,floor(.1*n)),:);
            [dump, center] = litekmeans(Xsubset, k, varargin{:}, 'start','sample', 'replicates',1 ,'MaxIter',clustermaxit);
        case 'numeric'
    end
    
    last = 0;label=1;
    it=0;
    
    switch distance
        case 'sqeuclidean'
            while any(label ~= last) && it<maxit
                last = label;
                
                bb = full(sum(center.*center,2)');
                ab = full(X*center');
                D = bb(ones(1,n),:) - 2*ab;
                
                [val,label] = min(D,[],2); % assign samples to the nearest centers
                ll = unique(label);
                if length(ll) < k
                    %disp([num2str(k-length(ll)),' clusters dropped at iter ',num2str(it)]);
                    missCluster = 1:k;
                    missCluster(ll) = [];
                    missNum = length(missCluster);
                    
                    aa = sum(X.*X,2);
                    val = aa + val;
                    [dump,idx] = sort(val,1,'descend');
                    label(idx(1:missNum)) = missCluster;
                end
                E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
                center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X);    % compute center of each cluster
                it=it+1;
            end
            if it<maxit
                bCon = true;
            end
            if isempty(bestlabel)
                bestlabel = label;
                bestcenter = center;
                if reps>1
                    if it>=maxit
                        aa = full(sum(X.*X,2));
                        bb = full(sum(center.*center,2));
                        ab = full(X*center');
                        D = bsxfun(@plus,aa,bb') - 2*ab;
                        D(D<0) = 0;
                    else
                        aa = full(sum(X.*X,2));
                        D = aa(:,ones(1,k)) + D;
                        D(D<0) = 0;
                    end
                    D = sqrt(D);
                    for j = 1:k
                        sumD(j) = sum(D(label==j,j));
                    end
                    bestsumD = sumD;
                    bestD = D;
                end
            else
                if it>=maxit
                    aa = full(sum(X.*X,2));
                    bb = full(sum(center.*center,2));
                    ab = full(X*center');
                    D = bsxfun(@plus,aa,bb') - 2*ab;
                    D(D<0) = 0;
                else
                    aa = full(sum(X.*X,2));
                    D = aa(:,ones(1,k)) + D;
                    D(D<0) = 0;
                end
                D = sqrt(D);
                for j = 1:k
                    sumD(j) = sum(D(label==j,j));
                end
                if sum(sumD) < sum(bestsumD)
                    bestlabel = label;
                    bestcenter = center;
                    bestsumD = sumD;
                    bestD = D;
                end
            end
        case 'cosine'
            while any(label ~= last) && it<maxit
                last = label;
                W=full(X*center');
                [val,label] = max(W,[],2); % assign samples to the nearest centers
                ll = unique(label);
                if length(ll) < k
                    missCluster = 1:k;
                    missCluster(ll) = [];
                    missNum = length(missCluster);
                    [dump,idx] = sort(val);
                    label(idx(1:missNum)) = missCluster;
                end
                E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
                center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X);    % compute center of each cluster
                centernorm = sqrt(sum(center.^2, 2));
                center = center ./ centernorm(:,ones(1,p));
                it=it+1;
            end
            if it<maxit
                bCon = true;
            end
            if isempty(bestlabel)
                bestlabel = label;
                bestcenter = center;
                if reps>1
                    if any(label ~= last)
                        W=full(X*center');
                    end
                    D = 1-W;
                    for j = 1:k
                        sumD(j) = sum(D(label==j,j));
                    end
                    bestsumD = sumD;
                    bestD = D;
                end
            else
                if any(label ~= last)
                    W=full(X*center');
                end
                D = 1-W;
                for j = 1:k
                    sumD(j) = sum(D(label==j,j));
                end
                if sum(sumD) < sum(bestsumD)
                    bestlabel = label;
                    bestcenter = center;
                    bestsumD = sumD;
                    bestD = D;
                end
            end
    end
end

label = bestlabel;
center = bestcenter;
if reps>1
    sumD = bestsumD;
    D = bestD;
elseif nargout > 3
    switch distance
        case 'sqeuclidean'
            if it>=maxit
                aa = full(sum(X.*X,2));
                bb = full(sum(center.*center,2));
                ab = full(X*center');
                D = bsxfun(@plus,aa,bb') - 2*ab;
                D(D<0) = 0;
            else
                aa = full(sum(X.*X,2));
                D = aa(:,ones(1,k)) + D;
                D(D<0) = 0;
            end
            D = sqrt(D);
        case 'cosine'
            if it>=maxit
                W=full(X*center');
            end
            D = 1-W;
    end
    for j = 1:k
        sumD(j) = sum(D(label==j,j));
    end
end
end

function [eid,emsg,varargout]=getargs(pnames,dflts,varargin)
%GETARGS Process parameter name/value pairs 
%   [EID,EMSG,A,B,...]=GETARGS(PNAMES,DFLTS,'NAME1',VAL1,'NAME2',VAL2,...)
%   accepts a cell array PNAMES of valid parameter names, a cell array
%   DFLTS of default values for the parameters named in PNAMES, and
%   additional parameter name/value pairs.  Returns parameter values A,B,...
%   in the same order as the names in PNAMES.  Outputs corresponding to
%   entries in PNAMES that are not specified in the name/value pairs are
%   set to the corresponding value from DFLTS.  If nargout is equal to
%   length(PNAMES)+1, then unrecognized name/value pairs are an error.  If
%   nargout is equal to length(PNAMES)+2, then all unrecognized name/value
%   pairs are returned in a single cell array following any other outputs.
%
%   EID and EMSG are empty if the arguments are valid.  If an error occurs,
%   EMSG is the text of an error message and EID is the final component
%   of an error message id.  GETARGS does not actually throw any errors,
%   but rather returns EID and EMSG so that the caller may throw the error.
%   Outputs will be partially processed after an error occurs.
%
%   This utility can be used for processing name/value pair arguments.
%
%   Example:
%       pnames = {'color' 'linestyle', 'linewidth'}
%       dflts  = {    'r'         '_'          '1'}
%       varargin = {{'linew' 2 'nonesuch' [1 2 3] 'linestyle' ':'}
%       [eid,emsg,c,ls,lw] = statgetargs(pnames,dflts,varargin{:})    % error
%       [eid,emsg,c,ls,lw,ur] = statgetargs(pnames,dflts,varargin{:}) % ok

% We always create (nparams+2) outputs:
%    one each for emsg and eid
%    nparams varargs for values corresponding to names in pnames
% If they ask for one more (nargout == nparams+3), it's for unrecognized
% names/values

%   Original Copyright 1993-2008 The MathWorks, Inc. 
%   Modified by Deng Cai (dengcai@gmail.com) 2011.11.27




% Initialize some variables
emsg = '';
eid = '';
nparams = length(pnames);
varargout = dflts;
unrecog = {};
nargs = length(varargin);

% Must have name/value pairs
if mod(nargs,2)~=0
    eid = 'WrongNumberArgs';
    emsg = 'Wrong number of arguments.';
else
    % Process name/value pairs
    for j=1:2:nargs
        pname = varargin{j};
        if ~ischar(pname)
            eid = 'BadParamName';
            emsg = 'Parameter name must be text.';
            break;
        end
        i = strcmpi(pname,pnames);
        i = find(i);
        if isempty(i)
            % if they've asked to get back unrecognized names/values, add this
            % one to the list
            if nargout > nparams+2
                unrecog((end+1):(end+2)) = {varargin{j} varargin{j+1}};
                % otherwise, it's an error
            else
                eid = 'BadParamName';
                emsg = sprintf('Invalid parameter name:  %s.',pname);
                break;
            end
        elseif length(i)>1
            eid = 'BadParamName';
            emsg = sprintf('Ambiguous parameter name:  %s.',pname);
            break;
        else
            varargout{i} = varargin{j+1};
        end
    end
end

varargout{nparams+1} = unrecog;
end

function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end

if nargin < 3
    isMax = 1;
    isSym = 1;
end

if nargin < 4
    isSym = 1;
end

if isSym == 1
    A = max(A,A');
end
[v d] = eig(A);
d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);
else
    [d1, idx] = sort(d,'descend');
end

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);
end

function [x ft] = EProjSimplex_new(v, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=1
%

if nargin < 2
    k = 1;
end

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end
    end
    x = max(v1,0);

else
    x = v0;
end
end

function [D,I] = pdist2(X,Y,dist,varargin)
%PDIST2 Pairwise distance between two sets of observations.
%   D = PDIST2(X,Y) returns a matrix D containing the Euclidean distances
%   between each pair of observations in the MX-by-N data matrix X and
%   MY-by-N data matrix Y. Rows of X and Y correspond to observations,
%   and columns correspond to variables. D is an MX-by-MY matrix, with the
%   (I,J) entry equal to distance between observation I in X and
%   observation J in Y.
%
%   D = PDIST2(X,Y,DISTANCE) computes D using DISTANCE.  Choices are:
%
%       'euclidean'        - Euclidean distance (default)
%       'squaredeuclidean' - Squared Euclidean distance
%       'seuclidean'       - Standardized Euclidean distance. Each
%                            coordinate difference between rows in X and Y
%                            is scaled by dividing by the corresponding
%                            element of the standard deviation computed
%                            from X, S=NANSTD(X). To specify another value
%                            for S, use
%                            D = PDIST2(X,Y,'seuclidean',S).
%       'cityblock'        - City Block distance
%       'minkowski'        - Minkowski distance. The default exponent is 2.
%                            To specify a different exponent, use
%                            D = PDIST2(X,Y,'minkowski',P), where the
%                            exponent P is a scalar positive value.
%       'chebychev'        - Chebychev distance (maximum coordinate
%                            difference)
%       'mahalanobis'      - Mahalanobis distance, using the sample
%                            covariance of X as computed by NANCOV.  To
%                            compute the distance with a different
%                            covariance, use
%                            D = PDIST2(X,Y,'mahalanobis',C), where the
%                            matrix C is symmetric and positive definite.
%       'cosine'           - One minus the cosine of the included angle
%                            between observations (treated as vectors)
%       'correlation'      - One minus the sample linear correlation
%                            between observations (treated as sequences of
%                            values).
%       'spearman'         - One minus the sample Spearman's rank
%                            correlation between observations (treated as
%                            sequences of values)
%       'hamming'          - Hamming distance, percentage of coordinates
%                            that differ
%       'jaccard'          - One minus the Jaccard coefficient, the
%                            percentage of nonzero coordinates that differ
%       function           - A distance function specified using @, for
%                            example @DISTFUN
%
%   A distance function must be of the form
%
%         function D2 = DISTFUN(ZI,ZJ),
%
%   taking as arguments a 1-by-N vector ZI containing a single observation
%   from X or Y, an M2-by-N matrix ZJ containing multiple observations from
%   X or Y, and returning an M2-by-1 vector of distances D2, whose Jth
%   element is the distance between the observations ZI and ZJ(J,:).
%
%   For built-in distance metrics, the distance between observation I in X
%   and observation J in Y will be NaN if observation I in X or observation
%   J in Y contains NaNs.
%
%   D = PDIST2(X,Y,DISTANCE,'Smallest',K) returns a K-by-MY matrix D
%   containing the K smallest pairwise distances to observations in X for
%   each observation in Y. PDIST2 sorts the distances in each column of D
%   in ascending order. D = PDIST2(X,Y,DISTANCE, 'Largest',K) returns the K
%   largest pairwise distances sorted in descending order. If K is greater
%   than MX, PDIST2 returns an MX-by-MY distance matrix. For each
%   observation in Y, PDIST2 finds the K smallest or largest distances by
%   computing and comparing the distance values to all the observations in
%   X.
%
%   [D,I] = PDIST2(X,Y,DISTANCE,'Smallest',K) returns a K-by-MY matrix I
%   containing indices of the observations in X corresponding to the K
%   smallest pairwise distances in D. [D,I] = PDIST2(X,Y,DISTANCE,
%   'Largest',K) returns indices corresponding to the K largest pairwise
%   distances.
%
%   Example:
%      % Compute the ordinary Euclidean distance
%      X = randn(100, 5);
%      Y = randn(25, 5);
%      D = pdist2(X,Y,'euclidean');         % euclidean distance
%
%      % Compute the Euclidean distance with each coordinate difference
%      % scaled by the standard deviation
%      Dstd = pdist2(X,Y,'seuclidean');
%
%      % Use a function handle to compute a distance that weights each
%      % coordinate contribution differently.
%      Wgts = [.1 .3 .3 .2 .1];            % coordinate weights
%      weuc = @(XI,XJ,W)(sqrt(bsxfun(@minus,XI,XJ).^2 * W'));
%      Dwgt = pdist2(X,Y, @(Xi,Xj) weuc(Xi,Xj,Wgts));
%
%   See also PDIST, KNNSEARCH, CREATENS, KDTreeSearcher,
%            ExhaustiveSearcher.

%   An example of distance for data with missing elements:
%
%      X = randn(100, 5);     % some random points
%      Y = randn(25, 5);      % some more random points
%      X(unidrnd(prod(size(X)),1,20)) = NaN; % scatter in some NaNs
%      Y(unidrnd(prod(size(Y)),1,5)) = NaN; % scatter in some NaNs
%      D = pdist2(X, Y, @naneucdist);
%
%      function D = naneucdist(XI, YJ) % euclidean distance, ignoring NaNs
%      [m,p] = size(YJ);
%      sqdxy = bsxfun(@minus,XI,YJ) .^ 2;
%      pstar = sum(~isnan(sqdxy),2); % correction for missing coordinates
%      pstar(pstar == 0) = NaN;
%      D = sqrt(nansum(sqdxy,2) .* p ./ pstar);
%
%
%   For a large number of observations, it is sometimes faster to compute
%   the distances by looping over coordinates of the data (though the code
%   is more complicated):
%
%      function D = nanhamdist(XI, YJ) % hamming distance, ignoring NaNs
%      [m,p] = size(YJ);
%      nesum = zeros(m,1);
%      pstar = zeros(m,1);
%      for q = 1:p
%          notnan = ~(isnan((XI(q)) | isnan(YJ(:,q)));
%          nesum = nesum + (XI(q) ~= YJ(:,q)) & notnan;
%          pstar = pstar + notnan;
%      end
%      D = nesum ./ pstar;

%   Copyright 2009-2018 The MathWorks, Inc.

if nargin > 2
    dist = convertStringsToChars(dist);
end

if nargin > 3
    [varargin{:}] = convertStringsToChars(varargin{:});
end

if nargin < 2
    error(message('stats:pdist2:TooFewInputs'));
end

if ~ismatrix(X) || ~ismatrix(Y)
    error(message('stats:pdist2:UnsupportedND'));
end

[nx,p] = size(X);
[ny,py] = size(Y);
if py ~= p
    error(message('stats:pdist2:SizeMismatch'));
end

additionalArg = [];

if nargin < 3
    dist = 'euc';
else %distance is provided
    if ischar(dist)
        methods = {'euclidean'; 'seuclidean'; 'cityblock'; 'chebychev'; ...
            'mahalanobis'; 'minkowski'; 'cosine'; 'correlation'; ...
            'spearman'; 'hamming'; 'jaccard'; 'squaredeuclidean'};
        i = find(strncmpi(dist,methods,length(dist)));
        if length(i) > 1
            error(message('stats:pdist2:AmbiguousDistance', dist));
        elseif isempty(i)
            error(message('stats:pdist2:UnrecognizedDistance', dist));
        else
            if i == 12 %'squaredeuclidean'
                dist = 'sqe'; % represent squared Euclidean
            else
                dist = methods{i}(1:3);
            end
            
            if ~isempty(varargin)
                arg = varargin{1};
                
                % Get the additional distance argument from the inputs
                if isnumeric(arg)
                    switch dist
                        case {'seu' 'mah' 'min'}
                            additionalArg = arg;
                            varargin = varargin(2:end);
                    end
                end
            end
        end
    elseif isa(dist, 'function_handle')
        distfun = dist;
        dist = 'usr';
    else
        error(message('stats:pdist2:BadDistance'));
    end
end

pnames = {'smallest' 'largest' 'radius' 'sortindices'};
dflts =  {        []        []  [] true};
[smallest,largest,radius,doSort] = internal.stats.parseArgs(pnames, dflts, varargin{:});

validateattributes(doSort,{'logical','numeric'},{'scalar'},'','sortindices');
doSort = logical(doSort);

smallestLargestFlag = [];
if sum([~isempty(largest) ~isempty(smallest) ~isempty(radius)]) > 1
    error(message('stats:pdist2:SmallestAndLargest'));
end
if ~isempty(smallest)
    if ~(isscalar(smallest) && isnumeric(smallest) && smallest >= 1 && round(smallest) == smallest)
        error(message('stats:pdist2:BadSmallest'));
    end
    smallestLargestFlag = min(smallest,nx);
elseif ~isempty(largest)
    if ~(isscalar(largest) && isnumeric(largest) && largest >= 1 && round(largest) == largest)
        error(message('stats:pdist2:BadLargest'));
    end
    smallestLargestFlag = -min(largest,nx);
elseif ~isempty(radius)
    if ~(isscalar(radius) && isnumeric(radius) && radius >= 0 )
        error(message('stats:pdist2:BadRadius'));
    end
elseif nargout > 1
    error(message('stats:pdist2:TooManyOutputs'));
end

% For a built-in distance, integer/logical/char/anything data will be
% converted to float. Complex floating point data can't be handled by
% a built-in distance function.
%if ~strcmp(dist,'usr')

try
    outClass = superiorfloat(X,Y);
catch
    if isfloat(X)
        outClass = class(X);
    elseif isfloat(Y)
        outClass = class(Y);
    else
        outClass = 'double';
    end
end

if ~strcmp(dist,'usr')
    % Built-in distances do not work for string arrays, so convert those to
    % char and see if that will work.
    X = convertStringsToChars(X);
    Y = convertStringsToChars(Y);
    if iscellstr(X)
        X = char(X);
    end
    if iscellstr(Y)
        Y = char(Y);
    end

    if ~strcmp(class(X),outClass) || ~strcmp(class(Y),outClass)
        warning(message('stats:pdist2:DataConversion', outClass));
    end
    X = cast(X,outClass);
    Y = cast(Y,outClass);
    if  ~isreal(X) || ~isreal(Y)
        error(message('stats:pdist2:ComplexData'));
    end
end

% Degenerate case, just return an empty of the proper size.
if (nx == 0) || (ny == 0)
    if ~isempty(radius)
        D = repmat({zeros(1,0, outClass)},1,ny);
        I = repmat({zeros(1,0)},1,ny);
    else
        if ~isempty(smallestLargestFlag)
            nD = abs(smallestLargestFlag);
        else
            nD = nx;
        end
        D = zeros(nD,ny,outClass); % X and Y were single/double, or cast to double
        I = zeros(nD,ny);
    end
    return;
end

switch dist
    case 'seu' % Standardized Euclidean weights by coordinate variance
        if isempty(additionalArg)
            additionalArg =  nanvar(X,[],1);
            if any(additionalArg == 0)
                warning(message('stats:pdist2:ConstantColumns'));
            end
            additionalArg = 1./ additionalArg;
        else
            if ~(isvector(additionalArg) && length(additionalArg) == p...
                    && all(additionalArg >= 0))
                error(message('stats:pdist2:InvalidWeights'));
            end
            if any(additionalArg == 0)
                warning(message('stats:pdist2:ZeroInverseWeights'));
            end
            additionalArg = 1./ (additionalArg .^2);
        end
        
    case 'mah' % Mahalanobis
        if isempty(additionalArg)
            if nx == 1
                error(message('stats:pdist2:tooFewXRowsForMah'));
            end
            additionalArg = nancov(X);
            [T,flag] = chol(additionalArg);
        else %provide the covariance for mahalanobis
            if ~isequal(size(additionalArg),[p,p])
                error(message('stats:pdist2:InvalidCov'));
            end
            %cholcov will check whether the covariance is symmetric
            [T,flag] = cholcov(additionalArg,0);
        end
        
        if flag ~= 0
            error(message('stats:pdist2:InvalidCov'));
        end
        
        if ~issparse(X) && ~issparse(Y)
            additionalArg = T \ eye(p); %inv(T)
        end
        
    case 'min' % Minkowski
        if isempty(additionalArg)
            additionalArg = 2;
        elseif ~(isscalar(additionalArg) && additionalArg > 0)
            error(message('stats:pdist2:BadMinExp'));
        end
    case 'cos' % Cosine
        [X,Y,flag] = normalizeXY(X,Y);
        if flag
            warning(message('stats:pdist2:ZeroPoints'));
        end
        
    case 'cor' % Correlation
        X = bsxfun(@minus,X,mean(X,2));
        Y = bsxfun(@minus,Y,mean(Y,2));
        [X,Y,flag] = normalizeXY(X,Y);
        if flag
            warning(message('stats:pdist2:ConstantPoints'));
        end
        
    case 'spe'
        X = tiedrank(X')'; % treat rows as a series
        Y = tiedrank(Y')';
        X = X - (p+1)/2; % subtract off the (constant) mean
        Y = Y - (p+1)/2;
        [X,Y,flag] = normalizeXY(X,Y);
        if flag
            warning(message('stats:pdist2:TiedPoints'));
        end
        
    otherwise
        
end

% Note that if the above switch statement is modified to include the
% 'che', 'euc', or 'cit' distances, that code may need to be repeated
% in the corresponding block below.
if strcmp(dist,'min') % Minkowski distance
    if isinf(additionalArg) %the exponent is inf
        dist = 'che';
        additionalArg = [];
    elseif additionalArg == 2 %the exponent is 2
        dist = 'euc';
        additionalArg = [];
    elseif additionalArg == 1 %the exponent is 1
        dist = 'cit';
        additionalArg = [];
    end
end

% Call a mex file to compute distances for the build-in distance measures
% on non-sparse real float (double or single) data.
if ~strcmp(dist,'usr') && (~issparse(X) && ~issparse(Y))
    additionalArg = cast(additionalArg,outClass);
    if strcmp(dist,'sqe')
       radius = sqrt(radius); 
    end
    if nargout < 2
        D = internal.stats.pdist2mex(X',Y',dist,additionalArg,smallestLargestFlag,radius,doSort);
    else
        [D,I] = internal.stats.pdist2mex(X',Y',dist,additionalArg,smallestLargestFlag,radius,doSort);
    end
    
    % The following MATLAB code implements the same distance calculations as
    % the mex file. It assumes X and Y are real single or double.  It is
    % currently only called for sparse inputs, but it may also be useful as a
    % template for customization.
elseif ~strcmp(dist,'usr')
    if any(strcmp(dist, {'ham' 'jac' 'che'}))
        xnans = any(isnan(X),2);
        ynans = any(isnan(Y),2);
    end
    
    if ~isempty(radius)
        D = cell(1,ny);
        I = cell(1,ny);
    elseif isempty(smallestLargestFlag)
        D = zeros(nx,ny,outClass);
    else
        D = zeros(abs(smallestLargestFlag),ny,outClass);
        I = zeros(abs(smallestLargestFlag),ny);
        
    end
    
    switch dist
        case 'euc'  % Euclidean
            if isempty(radius) && isempty(smallestLargestFlag) && (issparse(X) && issparse(Y))
                D = internal.stats.pdist2SparseMEX(X',Y',dist,additionalArg);
            else
                for i = 1:ny
                    dsq = zeros(nx,1,outClass);
                    for q = 1:p
                        dsq = dsq + (X(:,q) - Y(i,q)).^2;
                    end
                    dsq = sqrt(dsq);
                    if ~isempty(radius)
                        [D{i},I{i}] = radiusSort(dsq,radius);
                    elseif isempty(smallestLargestFlag)
                        D(:,i) = dsq;
                    else
                        [D(:,i),I(:,i)] = partialSort(dsq,smallestLargestFlag);
                    end
                end
            end
        case 'sqe'  % Squared Euclidean
            if isempty(radius) && isempty(smallestLargestFlag) && (issparse(X) && issparse(Y))
                D = internal.stats.pdist2SparseMEX(X',Y',dist,additionalArg);
            else
                for i = 1:ny
                    dsq = zeros(nx,1,outClass);
                    for q = 1:p
                        dsq = dsq + (X(:,q) - Y(i,q)).^2;
                    end
                    
                    if ~isempty(radius)
                        [D{i},I{i}] = radiusSort(dsq,radius);
                    elseif isempty(smallestLargestFlag)
                        D(:,i) = dsq;
                    else
                        [D(:,i),I(:,i)] = partialSort(dsq,smallestLargestFlag);
                    end
                end
            end
        case 'seu'    % Standardized Euclidean
            wgts = additionalArg;
            for i = 1:ny
                dsq = zeros(nx,1,outClass);
                for q = 1:p
                    dsq = dsq + wgts(q) .* (X(:,q) - Y(i,q)).^2;
                end
                dsq = sqrt(dsq);
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(dsq,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) = dsq;
                else
                    [D(:,i),I(:,i)] = partialSort(dsq,smallestLargestFlag);
                end
            end
            
        case 'cit'    % City Block
            for i = 1:ny
                dsq = zeros(nx,1,outClass);
                for q = 1:p
                    dsq = dsq + abs(X(:,q) - Y(i,q));
                end
                
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(dsq,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) = dsq;
                else
                    [D(:,i),I(:,i)] = partialSort(dsq,smallestLargestFlag);
                end
            end
            
        case 'mah'    % Mahalanobis
            for i = 1:ny
                del = bsxfun(@minus,X,Y(i,:));
                dsq = sum((del/T) .^ 2, 2);
                dsq = sqrt(dsq);
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(dsq,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) = dsq;
                else
                    [D(:,i),I(:,i)] = partialSort(dsq,smallestLargestFlag);
                end
            end
            
        case 'min'    % Minkowski
            expon = additionalArg;
            for i = 1:ny
                dpow = zeros(nx,1,outClass);
                for q = 1:p
                    dpow = dpow + abs(X(:,q) - Y(i,q)).^expon;
                end
                dpow = dpow .^ (1./expon);
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(dpow,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) = dpow;
                else
                    [D(:,i),I(:,i)] = partialSort(dpow,smallestLargestFlag);
                end
                
            end
            
        case {'cos' 'cor' 'spe'}   % Cosine, Correlation, Rank Correlation
            % This assumes that data have been appropriately preprocessed
            for i = 1:ny
                d = zeros(nx,1,outClass);
                for q = 1:p
                    d = d + (X(:,q).*Y(i,q));
                end
                d(d>1) = 1; % protect against round-off, don't overwrite NaNs
                d = 1 - d;
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(d,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) = d;
                else
                    [D(:,i),I(:,i)] = partialSort(d,smallestLargestFlag);
                end
                
            end
        case 'ham'    % Hamming
            for i = 1:ny
                nesum = zeros(nx,1,outClass);
                for q = 1:p
                    nesum = nesum + (X(:,q) ~= Y(i,q));
                end
                nesum(xnans|ynans(i)) = NaN;
                nesum = (nesum ./ p);
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(nesum,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) = nesum;
                else
                    [D(:,i),I(:,i)] = partialSort(nesum,smallestLargestFlag);
                end
                
            end
        case 'jac'    % Jaccard
            for i = 1:ny
                nzsum = zeros(nx,1,outClass);
                nesum = zeros(nx,1,outClass);
                for q = 1:p
                    nz = (X(:,q) ~= 0 | Y(i,q) ~= 0);
                    ne = (X(:,q) ~= Y(i,q));
                    nzsum = nzsum + nz;
                    nesum = nesum + (nz & ne);
                end
                nesum(xnans | ynans(i)) = NaN;
                d = (nesum ./ nzsum);
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(d,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) = d;
                else
                    [D(:,i),I(:,i)] = partialSort(d,smallestLargestFlag);
                end
                
            end
        case 'che'    % Chebychev
            for i = 1:ny
                dmax = zeros(nx,1,outClass);
                for q = 1:p
                    dmax = max(dmax, abs(X(:,q) - Y(i,q)));
                end
                dmax(xnans | ynans(i)) = NaN;
                if ~isempty(radius)
                    [D{i},I{i}] = radiusSort(dmax,radius);
                elseif isempty(smallestLargestFlag)
                    D(:,i) =  dmax;
                else
                    [D(:,i),I(:,i)] = partialSort(dmax,smallestLargestFlag);
                end
            end
    end
    
    % Compute distances for a caller-defined distance function.
else % if strcmp(dist,'usr')
    try
        D = feval(distfun,Y(1,:),X(1,:));
    catch ME
        if strcmp('MATLAB:UndefinedFunction', ME.identifier) ...
                && ~isempty(strfind(ME.message, func2str(distfun)))
            error(message('stats:pdist2:DistanceFunctionNotFound', func2str( distfun )));
        end
        % Otherwise, let the catch block below generate the error message
        D = [];
    end
    
    if ~isnumeric(D)
        error(message('stats:pdist2:OutputBadType'));
    end
    
    if ~isempty(radius)
        D = cell(1,ny);
        if nargout >= 2
            I = cell(1,ny);
        end
        
        for i = 1:ny
            try
                temp = feval(distfun,Y(i,:),X);
            catch ME
                if isa(distfun, 'inline')
                    m = message('stats:pdist2:DistanceInlineError');
                    ME2 = MException(m.Identifier,'%s',getString(m));
                    throw(addCause(ME2,ME));
                else
                    m = message('stats:pdist2:DistanceFunctionError',func2str(distfun));
                    ME2 = MException(m.Identifier,'%s',getString(m));
                    throw(addCause(ME2,ME));
                end
            end
            
            if nargout < 2
                D{i} = radiusSort(temp,radius);
            else
                [D{i},I{i}] = radiusSort(temp, radius);
            end
        end
    elseif ~isempty(smallestLargestFlag)
        D = zeros(abs(smallestLargestFlag),ny,class(D));
        if nargout > 1
            I = zeros(abs(smallestLargestFlag),ny,class(D));
        end
        
        for i = 1:ny
            
            try
                temp = feval(distfun,Y(i,:),X);
            catch ME
                if isa(distfun, 'inline')
                    m = message('stats:pdist2:DistanceInlineError');
                    ME2 = MException(m.Identifier,'%s',getString(m));
                    throw(addCause(ME2,ME));
                else
                    m = message('stats:pdist2:DistanceFunctionError',func2str(distfun));
                    ME2 = MException(m.Identifier,'%s',getString(m));
                    throw(addCause(ME2,ME));
                end
            end
            
            if nargout < 2
                D(:,i) = partialSort(temp,smallestLargestFlag);
            else
                [D(:,i),I(:,i)] = partialSort(temp,smallestLargestFlag);
            end
        end
        
    else  %compute all the pairwise distance
        % Make the return have whichever numeric type the distance function
        % returns.
        D = zeros(nx,ny,class(D));
        
        for i = 1:ny
            try
                D(:,i) = feval(distfun,Y(i,:),X);
                
            catch ME
                if isa(distfun, 'inline')
                    m = message('stats:pdist2:DistanceInlineError');
                    ME2 = MException(m.Identifier,'%s',getString(m));
                    throw(addCause(ME2,ME));
                else
                    m = message('stats:pdist2:DistanceFunctionError',func2str(distfun));
                    ME2 = MException(m.Identifier,'%s',getString(m));
                    throw(addCause(ME2,ME));
                end
            end
        end
    end
    
end
end
function [X,Y,flag] = normalizeXY(X,Y)
Xmax = max(abs(X),[],2);
X2 = bsxfun(@rdivide,X,Xmax);
Xnorm = sqrt(sum(X2.^2, 2));

Ymax = max(abs(Y),[],2);
Y2 = bsxfun(@rdivide,Y,Ymax);
Ynorm = sqrt(sum(Y2.^2, 2));
% Find out points for which distance cannot be computed.

% The norm will be NaN for rows that are all zeros, fix that for the test
% below.
Xnorm(Xmax==0) = 0;
Ynorm(Ymax==0) = 0;

% The norm will be NaN for rows of X that have any +/-Inf. Those should be
% Inf, but leave them as is so those rows will not affect the test below.
% The points can't be normalized, so any distances from them will be NaN
% anyway.

% Find points that are effectively zero relative to the point with largest norm.
flag =  any(Xnorm <= eps(max(Xnorm))) || any(Ynorm <= eps(max(Ynorm)));
Xnorm = Xnorm .* Xmax;
Ynorm = Ynorm .* Ymax;
X = bsxfun(@rdivide,X,Xnorm);
Y = bsxfun(@rdivide,Y,Ynorm);
end
function [D,I] = partialSort(D,smallestLargest)
if smallestLargest > 0
    n = smallestLargest;
else
    %sort(D,'descend') puts the NaN values at the beginning of the sorted list.
    %That is not what we want here.
    D = D*-1;
    n = -smallestLargest;
end

if nargout < 2
    D = sort(D,1);
    D = D(1:n,:);
else
    [D,I] = sort(D,1);
    D = D(1:n,:);
    I = I(1:n,:);
end

if smallestLargest < 0
    D = D * -1;
end
end
function [D,I] = radiusSort(D,radius)
I = find (D <= radius);
D = D(I);
if nargout < 2
    D = sort(D,1)'; %return a row vector
else
    [D,I2] = sort(D,1);
    D= D';
    I = I(I2)';
end
end

% min_{S>=0, S'*1=1, S*1=1, F'*F=I}  ||S - A||^2 + 2*lambda*trace(F'*Ln*F)
function [y1,y2,SS,U,V,cs] = coclustering_bipartite_fast1(A, c, NITER,distX, alpha,islocal)

if nargin < 6
    islocal = 0;
end

if nargin < 5
    NITER = 30;
end

zr = 10e-11;
lambda = 0.1;

[n,m] = size(A);
onen = 1/n*ones(n,1);
onem = 1/m*ones(m,1);

A = sparse(A);
a1 = sum(A,2);
D1a = spdiags(1./sqrt(a1),0,n,n); 
a2 = sum(A,1);
D2a = spdiags(1./sqrt(a2'),0,m,m); 
A1 = D1a*A*D2a;

SS2 = A1'*A1; 
SS2 = full(SS2);

% automatically determine the cluster number
[V, ev0, ev]=eig1(SS2,m); 
aa = abs(ev); aa(aa>1-zr)=1-eps;
ad1 = aa(2:end)./aa(1:end-1);
ad1(ad1<0.15)=0; ad1 = ad1-eps*(1:m-1)'; ad1(1)=1;
ad1 = 1 - ad1;
[scores, cs] = sort(ad1,'descend');
cs = [cs, scores];

if nargin == 1
    c = cs(1);
end

V = V(:,1:c); 
U=(A1*V)./(ones(n,1)*sqrt(ev0(1:c)'));
U = sqrt(2)/2*U; V = sqrt(2)/2*V;  
a(:,1) = ev;
A = full(A); 
idxa = cell(n,1);
for i=1:n
    if islocal == 1
        idxa0 = find(A(i,:)>0);
    else
        idxa0 = 1:m;
    end
    idxa{i} = idxa0; 
end

idxam = cell(m,1);
for i=1:m
    if islocal == 1
        idxa0 = find(A(:,i)>0);
    else
        idxa0 = 1:n;
    end
    idxam{i} = idxa0; 
end

D1 = 1; D2 = 10;
for iter = 1:NITER
    
    U1 = D1*U;
    V1 = D2*V;
    dist = L2_distance_1(U1',V1');  
    
    S = zeros(n,m);
    for i=1:n
        idxa0 = idxa{i};
        dfi = dist(i,idxa0);
        dxi = distX(i,idxa0);
        ad = -(dxi+lambda*dfi)./(2*alpha);
        S(i,idxa0) = EProjSimplex_new(ad);
        

    end
    
    Sm = zeros(m,n);
    for i=1:m
        idxa0 = idxam{i};
        dfi = dist(idxa0,i);
        dxi = distX(idxa0,i);
        ad = -(dxi+lambda*dfi)./(2*alpha);
        Sm(i,idxa0) = EProjSimplex_new(ad);
    end

    S = sparse(S);
    Sm = sparse(Sm);    
    SS = (S+Sm')/2;
    d1 = sum(SS,2);
    D1 = spdiags(1./sqrt(d1),0,n,n);
    d2 = sum(SS,1);
    D2 = spdiags(1./sqrt(d2'),0,m,m);
    SS1 = D1*SS*D2;
    
    SS2 = SS1'*SS1;
    SS2 = full(SS2);
    [V, ev0, ev]=eig1(SS2,c);
    U=(SS1*V)./(ones(n,1)*sqrt(ev0'));
    U = sqrt(2)/2*U; V = sqrt(2)/2*V;
    
    
    a(:,iter+1) = ev;
    U_old = U;
    V_old = V;
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c)); 
    if fn1 < c-0.0000001
        lambda = 2*lambda;
    elseif fn2 > c-0.0000001
        lambda = lambda/2;   U = U_old; V = V_old; 
    else
        break;
    end
end

SS0=sparse(n+m,n+m); SS0(1:n,n+1:end)=SS; SS0(n+1:end,1:n)=SS';  
[clusternum, y]=graphconncomp(SS0);

y1=y(1:n)'; 
y2=y(n+1:end)'; 
end


