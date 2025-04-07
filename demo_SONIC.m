%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                               %
% This is a demo for the SONIC algorithm, which is proposed in the paper below. %
%                                                                               %
% Xian-Xian Xia, Dong Huang, Chen-Min Yang, Chaobo He, Chang-Dong Wang.         %
% Simple One-step Multi-view Clustering with Fast Similarity and Cluster        %
% Structure Learning. IEEE Signal Processing Letters, 2025.                     %
%                                                                               %
% The code has been tested in Matlab R2021a on a PC with Windows 10.            %
%                                                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function demo_SONIC()

clear;
close all;
clc;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           %% Load the data.
% Please modify the dataset name that you want to test.
% dataName = 'data_ALOI-100';
dataName = 'data_Out-Scene';
load([dataName,'.mat'],'fea','gt'); 
c = numel(unique(gt)); 
opts.Distance = 'sqEuclidean';

% Best parameters %

% Out-Scene:
m = 8;   % The number of anchors
alpha = 100; % The trade-off parameter

% ALOI-100: 
% m =100; 
% alpha = 0.001;

tic;
disp("Running SONIC...");
Label = SONIC(fea,c,m,alpha,opts);
disp("Done.");
toc;

disp('The NMI score on this dataset:')
scores = NMImax(Label,gt);
disp(['NMI = ',num2str(scores)]);
end