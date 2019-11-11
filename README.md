# trading_rl
The project contains a complete environment and agent to use Deep Reinforcement Learning for trading. The environment uses the Quant Connect API in Python 3

I always try to develope clean code and architecture that is readable for someone else. I do NOT use line breaks in my code as modern editors of IDEs can do it better for you...

`Data Ingestion / Creating datasets`

For data ingestion please make sure to install the LEAN environment first (https://github.com/QuantConnect/Lean). After installing LEAN and its dependencies, copy the QuantDataLoder notebook into your Lean/Launcher/bin/Debug directory and follow my documentation to create a Dataset object. 

`Agent`

The Agent is actually two separated agents: One to decide when to open a position, one Agent to decide when to close the position.

`OpeningAgent`

To open a position I try to approximize a Gaussian distribution (https://en.wikipedia.org/wiki/Normal_distribution) that returns the probability of making a positive return on invest. This comes along with the advantage of configure your risk at real-time! The agent doesn't use any state-of-the-art reinforcement-learning methods like TD-Learning methods or Policy Gradient methods. So, you will encounter some differences to (for example) OpenAI solutions. This is no gaming environment and your trading success is highly dependent on one wrong decision. For instance, to train the OpeningAgent, only the opening action (opening a long or short position) is taken into account.
Technically a number of previous bars of normalized MACD values is forwarded to a CNN followed by fully connected layers. The output of the network is a mean value (from -1 to 1) and a variance value.
Due to training stability reasons, the function to optimize is not the Gaussian Normal Distribution itself, but the natural logarithm of it:

![loss function](https://latex.codecogs.com/svg.latex?L&space;=&space;\frac{r-\mu}{\sigma^2}&space;&plus;&space;\frac{1}{2}\cdot&space;log(2\pi\sigma^2))

![loss derivatives](https://latex.codecogs.com/svg.latex?\frac{\partial&space;L}{\partial&space;\mu}=-\frac{1}{\sigma^2}&space;,&space;\frac{\partial&space;L}{\partial&space;\sigma^2}=\frac{1}{2\sigma^2}-\frac{r-\mu}{\sigma^4})

With the estimation of the mean and variance you can calculate the probability to gain a reward of more than 0 with the help of the cumulative distribution

![win probability](https://latex.codecogs.com/svg.latex?P_{win}&space;=&space;\frac{1}{2}&space;(1&space;&plus;&space;erf(\frac{-\mu}{\sqrt{2\sigma^2}})))

The reward is calculated with a cumulative weighted sum of each win per bar, beginning from the opening bar to an amount of bars in the future. The closer the bar to the opening, the heigher the weight.
