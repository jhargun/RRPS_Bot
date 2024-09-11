# Repeated Rock Paper Scissors Bot

## Introduction

This repo contains code used to train an agent to play repeated rock-paper-scissors against other agents. The goal is to play 1000 games of rock paper scissors and to learn how the other agent plays in order to beat them by winning at least 501 of the 1000 games.

A diverse range of agents served as opponents. Some were very simplistic (such as one which always played rock or another that just copied the opponent's last move). Others were much more complex, using techniques such as markov chains to try to model their opponent and win. These opponent agents were implemented by Google Deepmind's OpenSpiel package, which provides a framework for reinforcement learning.

## My Agent

If you don't want to read the in-depth explanation of my agent, feel free to skip to the summary at the end

### Beating Greenberg

I started with the goal of beating only Greenberg, the most advanced opponent. In other words, I wanted to make sure that when my agent and Greenberg played head to head, my agent had a positive average return. To do this, I first made Greenberg and randbot (a bot which plays random moves every time) play 3000 games against each other and saved the game histories. I then used these histories to train an LSTM to predict Greenberg’s next move. I went for this approach for 2 reasons. First, I knew LSTMs were fairly good at predicting future outcomes from past time series data. Second, I wasn’t able to get Openspiel working on my Windows desktop and I couldn’t train a model in Colab using GPU without getting kicked off part of the way through. Therefore, I decided to create all the data at the start using Colab (which, since I was using a CPU environment, didn’t kick me off when I ran data creation overnight) and then train an LSTM on it. The LSTM reached ~40% validation accuracy (I call this validation, not test, accuracy because it was not a holdout set and was used to select a model). This was better than random chance, so I thought it would be acceptable.

My first agent used the model’s predictions every time. When I tested this against Greenberg, it lost most of its games. After testing some more, I realized the agent did much better when a large proportion of its moves were random. I think this is because it was trained on random moves, so if the model is making most or all of the moves, the game starts to change too much from the data it was trained on, making the model’s predictions less reliable. After some testing, I found that a window size of 200 (meaning the LSTM looks at the previous 200 moves to make its predictions) and a random chance of .8 (meaning 80% of the moves are random) worked best.

### Beating all agents

This agent worked well, but it could only beat Greenberg. To generalize it, I first made data for the rest of the agents. To do this, I had each agent play against randbot for 1000 games each, again using Colab to run these games. I also altered my model to add the first 200 actions as a second input. My theory was that this could be used to help determine which of the bots it’s playing against and adjust the prediction accordingly. I used a regular, fully connected neural network instead of an LSTM since the purpose of this data was to determine the opponent, not make a prediction of the future.

When I first trained this agent, it had good accuracy, but this metric was misleading. Looking deeper, I found it was performing very well on the easier agents but had ~33% accuracy (as good as random chance) when predicting the output of harder agents. To fix this, I changed the loss function from categorical cross entropy to focal loss, which weights harder inputs higher, and trained for 20 epochs. As expected, this led to improved accuracy. I reused the window size of 200 (since I didn’t have time to retest other window sizes) and used a random chance of .8 (since this performed best on Greenberg, which I wanted to make sure I could still bea).

Although the randomness was optimized against Greenberg, my agent does well against most agents. When testing with 400 episodes, my agent had a >53% win rate against all bots except randbot, markov5, and phasenbott. Of these 3, the only one it lost consistently against was markov5 (against whom the win rate was 46%), though even then the average return of my agent was -.98, indicating that the games were close. 


### Summary

My first model focused on beating a single agent, Greenberg. I gathered data on Greenberg's playstyle by playing thousands of games where I made random moves against Greenberg. I then fed this data into an LSTM, which learned to predict Greenberg's moves with ~40% accuracy. My agent made random moves 80% of the time, while the other 20% of the time it used the LSTM to predict Greenberg's move and beat it. This successfully beat Greenberg.

My next model was designed to beat all opponents, not just Greenberg. I gathered data on all opponents the same way as before. Instead of just using an LSTM, I added a fully connected component to the model which tooki the first 200 moves of the game as input, allowing the model to predict its opponent and adjust its predictions accordingly. I also switched to focal loss. This model beat every opponent except randbot (which is impossible to reliably beat as it is completely random) and markov5 (which it still beat 47% of the time).

## Results and Code

The full results of the model trained against all opponents are shown in `results.txt`. To generate these results, I ran my agent against each opponent for a total of 400 games (each of which consists of 1000 rounds of rock paper scissors). This file contains the average return (meaning how much my model won/lost by on average), the number of total wins/losses/draws out of 400, and the resulting win rate of my model. A win rate over 50% means my model beat the opponent.

The evaluation script used is provided in `EvaluateAllBotsModel.ipynb` (though the notebook will only run on Colab since I was saving intermediate results to Google Drive to avoid losing the data). 

There are a few files containing code to train models, each corresponding to a different version of the model. Note that to run any of these, you will need to generate new data yourself.
- The code used to train the initial model (which was only trained against Greenberg) is provided in `lstm_torch_multiproc.py`
- The code to train the model against all opponents is in `allbots_lst_better.py`.
- The code for an older version of the model trained against all opponents (which did not use focal loss) is in `allbots_lstm.py`
- The code for an older version of the model trained against all opponents (which did not use an LSTM at all and performed very poorly) is in `allbots_noLSTM.py`.

The code to generate data is provided in `generateData.ipynb`, though I recommend against running it since it takes a lot of time to generate data.
