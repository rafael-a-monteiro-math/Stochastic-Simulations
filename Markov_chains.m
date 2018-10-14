%{
I would like to simulate a markov chain. I`ll implement the problem in two
ways:

1) using an algorithm that seems to be the most obvious

2) using an optimized version of the algorithm.

% keywords: imagesc
waterflow
%}

%%%% First test: filling matrix with colors
% 
% N_simulations = 100;
% states = 5;
% M = zeros(states,N_simulations);
% 
% figure()
% % every row has to add up to zero. the way we do that is the following
% colormap jet
% for (i = 1:N_simulations)
%     M(:,i) = rand(states,1);
%     M(:,i) = M(:,i)/sum(M(:,i));
%     f = imagesc(M);
%     pause(0.05);  
%     i
% end
% 
% close();

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

% As we want to simulate this Markov Chain, we do
% In this fashion, given a vector of probabilities, (p_1, ....p_N), we
% concatenate N intervals with legths p_1,,,p_N.
Distribute = ones(5,5);
Distribute = triu(Distribute);
Distributed_pts = M*Distribute;

%v = VideoWriter('Markov_chain.avi');
%open(v);

figure;
subplot(3,1,1);
colormap jet
imagesc(Phase_space);
title('particle state')

hold on;

plot_freq= subplot(3,1,2);
imagesc(frequency_vector);
title('Frequency');
hold on;
Phase_space(:,1) = state_vector';

% Since this is a Matrix..... the system will converge to its stationary
% distribution, which we can easily find numerically

subplot(3,1,3);
[V D] = eig(M');
stationary = V(:,4)/sum(V(:,4));
imagesc(real(stationary)')
title('stationary distribution')
colormap jet
hold on;


%In this manner, instead of worrying about the next state vector, we worry about the next interval
for (i = 2:N_simulations)
    Mov(i) = getframe(gcf) % leaving gcf out crops the frame in the movie
   
    next_interval_distribution = (Phase_space(:,i-1)')*Distributed_pts; 
    aux = (next_interval_distribution >= rand());
    state_vector = (1:5==find(aux,1, 'first')); % this is just a nice way to find the kth standard basis vector
    Phase_space(:,i) = state_vector';
    subplot(3,1,1);
    imagesc(Phase_space);
    pause(0.01)
    frequency_vector = sum(Phase_space(:,1:i)')/i;
    subplot(3,1,2);
    cla(plot_freq);
    imagesc(frequency_vector);
    
     writeVideo(v,M)
end

close();
close(v)
% The bad algorithm

%output the movie as an avia file