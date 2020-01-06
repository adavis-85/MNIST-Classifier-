
using Pkg
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")

using LinearAlgebra
using Statistics

##Loading MNIST dataset for training
Train_x,Train_y = MNIST.traindata()

##Loading MNIST dataset for testing
Test_x,Test_y=MNIST.testdata()

##For the basic least squares way we want to have the Test_x in matrix form not tensor which we will use for a following test .
Tr_x=(reshape(Train_x,28^2,60000))
Te_x=(reshape(Test_x,28^2,10000))

##Lets select the first image to look at for comparison on how the algorithm works
##p_img=zeros(size(Train_x,1),0)
imshow(Train_x[:,:,1])
Train_y[1]

##We are building a function which will return a matrix of zeros and ones with the ones being the position where the 
##specific class is found,zero being where it is not.
function one_hot(y,num_of_classes)
    
    A=zeros(length(y),num_of_classes)
    
    for i =1:class
        A[findall(y .==i),i] .=1
    end
    return A
end

##We also need to find a function which will return which class the image 
##most likely belongs to.  This location is shown in largest value of each row which 
##would be the strongest likelyhood of the class being the location.

max_location_in_row(A)=[argmax(A[i,:]) for i =1:size(A,1)]

##We need to now change the Train_x set into a matrix suitable for a least squares process.  
##We add a vector of ones to the beginning of the training set. 
##Also the same will be done with the training set.
Train_tr_x=hcat(ones(60000),Tr_x')
Test_te_x=hcat(ones(10000),Te_x')

##This is where we use the multiclass function to help us to find the parameters 
##suited for out least squares model.
function multi_class(x,y,class)
theta = x \ (2*Onehot(y,class) .- 1) 
parameters = max_location_in_row(x*theta)
return theta, parameters
end

T,P=multi_class(Train_tr_x,Train_y,10)

##Accuracy average for training set
mean(Train_y .!= P)

##Applying the parameters from our training set to the test set.
Test_value=Max_row(Test_te_x*T)

##Accuracy average for the test set.
mean(Test_y .!= Test_value)


