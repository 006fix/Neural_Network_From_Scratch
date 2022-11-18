
# back prop will have to apply ever N iterations of forward prop, where N = batch size
# it relies on knowing not only the weights and biases of every neuron, but also the values output for each input
#
#as such, we should run this EVERY TIME, store the values in each node, and per batch, calculate the true changes
    #apply these changes, and reset the node
#
#in terms of what we need to store an addition of the modifications for each weight and bias, and then average them
    #as such, we need a dict containing every weight and bias, and a counter that counts the number of additions made
#
#each node already stores a dict of its own forward weights, and its bias. we can take values from here
#
#STEP 1 - MAKE A NEW FUNCTION TO HOLD THESE VALUES ON THE NODES
    #complete - self.modification_dict, self.modification_count
#STEP 2 - MAKE A FUNCTION THAT POPULATES THESE VARIABLES WITH THE WEIGHT AND BIAS LABELS, SETS VALUE TO 0
    #THIS MUST RUN ON STARTUP, AND BE RERUN EVERY SINGLE TIME
    #to save effort here, make a function in the NN itself that sets these values to 0, then no need to run on startup
    #function can instead be rerun between batches
    #complete, see code

#STEP 1 AND 2 ARE NOW COMPLETE


#STEP 3 - DESIGN BACKWARD PROPOGATION FUNCTION.
#partial progress made. top layer complete.

#BUT HOW DO WE GET A GENERAL FORMULA FOR LOWER LAYERS?
#BASIC RULE OF THUMB - for every weight, this will be as follows
#2(ypred-yact) * preceeding weight * nodes activation function derivative * source of input
#but what happens when there's more layers in the way?
#2(ypred-yact)*preceeding weight * nodes activation function derivative * preceding weight * nodes activation function derivative * source of input
#now lets reverse, and generalise:
#lets assume we're trying to understand the modified weight of connection a1->b1.
#DUE TO THE BACKWARDS NATURE OF OUR APPROACH, WE ARE IN FACT AT THE STATE OF CHECKING NODE B AT THIS POINT
#in reality, a represents the input neuron to a neuron in the first layer
#past the first layer, b there is a hidden layer (c), and then finally the output layer (d)
#base multipler = source of input (input val)
#additional constant base = 2(ypred-yact)
#for current node
#check 'is output'
#if yes:
    #complete
#if no:
    #multiply by derivative of current node
    #look at node above.
    #is it the output?
    #if yes:
        #multiply by the weight of the connection from B to C, end
    #if no:
        #multiply by derivative of current node
        #look at node above
        #is it the output?
        #if yes:
            #multiply by the weight of the connection from C to D, end
        #if no:
            #rinse and repeat
###BUT WHAT IF ITS A BIAS FUNCTION?
#IN THIS CASE, IT IS VERY SIMILAR, HOWEVER,RATHER THAN MULTIPLYING BY THE INPUT VAL, YOU MULTIPLY BY 1
