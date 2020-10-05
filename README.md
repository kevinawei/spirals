# spirals

CS 101 MP 3

For the different parameters, I found that having the number of neurons for the first layer be about double the number of neurons for the second layer made the accuracy higher.
Making the number of neurons too great also increased the likelyhood that the model would overfit, but also made it so that it would train faster and reach a higher accuracy. 
Eventually I found the sweetspot to be around 220/110 neurons for layers 1 and 2 respectively.

Also, for the number of epochs, I first tried a lower number around 100, but found that it took my model much longer than that to achieve a desireable accuracy. 
I also found that for the learning rate, a lower number would make it less likely to overfit the model and keep the final accuracy higher

Adding a second layer also greatly increased the accuracy.

I tried using multiple activation functions such as sigmoid, but found that relu was ultimately the best for my model. 
