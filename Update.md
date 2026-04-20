### Current Updates

The model has completed training as of last week, but I couldnt find the time to test its performance on any data, synthetic or real. Real world data collection is about 5-10% complete only, with almost 100 Km curve unlabelled images
being the all of it. The testing phase is owing its delays mostly due to the lack of a good evaluation metric for the json data for now. I plan to search for a good approximation online, and if I cant find anything, then
Ill have to come up with something myself. If the model succeeds with synthetic data, and struggles with the real world, we will begin fine tuning on the real world dataset, after we complete it of course. If the results 
dont hold up, even on the synthetic data, we have two options, continue further training on the remaining images that include almost 80% of the 500k or so we initially geenrated. It might turn out to be a long grind.

The second option is that we come up with another strategy for how our loss is calculated during backpropagation, so we might give up on the current trajectory completely and pivot to Reinforcement learning, or we might have to
come up with an entirely new loss function. I know all of these options sound crazy, the latest one perhaps most of all, but nevertheless they are still options fro consideration given how the model might not be generalizing well on the current ADAM based optimizer.

I hope we clear at least the synthetic testing with grace so we have some ground to stand up for our next steps.
