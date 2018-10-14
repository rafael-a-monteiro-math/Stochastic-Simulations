%{
I would like to simulate a markov chain. I`ll implement the problem in two
ways:

1) using an algorithm that seems to be the most obvious

2) using an optimized version of the algorithm.

% keywords: imagesc
waterflow
%}

%%%% First test: filling matrix with colors
%First, I`ll start with a Markov 5X5 matrix

M = [.8,.2, 0,0,0; 
    .1,.8,.1,0,0; 
    0,.1,.8,.1,0;
    0 ,0,.1,.8,.1;
    0, 0,0,.2, .8 ];

state_vector = [1,0,0,0,0];% this is where we start

frequency_vector = state_vector;
%Now we do the simulation fow each state:
% in order to do so, we distribute the intervals in the space. 
N_simulations = 500;
Phase_space = zeros(5,N_simulations);

Phase_space(:,1) = state_vector'; %initial state

% As we want to simulate this Markov Chain, we do
% In this fashion, given a vector of probabilities, (p_1, ....p_N), we
% concatenate N intervals with legths p_1,,,p_N.
Distribute = ones(5,5);
Distribute = triu(Distribute);
Distributed_pts = M*Distribute;

v = VideoWriter('Markov_chain.avi');
open(v);

figure;
colormap jet
imagesc(Phase_space);
title('Markov Chains: simulation #1')
xlabel('Iterate')
ylabel('State')
yticks(1:5)

hold on;


%In this manner, instead of worrying about the next state vector, we worry about the next interval
for (i = 2:N_simulations)
    Mov = getframe(gcf) % leaving gcf out crops the frame in the movie
   
    next_interval_distribution = (Phase_space(:,i-1)')*Distributed_pts; 
    aux = (next_interval_distribution >= rand());
     state_vector = (1:5==find(aux,1, 'first')); % this is just a nice way to find the kth standard basis vector
    Phase_space(:,i) = state_vector';
    imagesc(Phase_space);
    pause(0.01)
    writeVideo(v,Mov)
end

close(v)
close();

%output the movie as an avia file
