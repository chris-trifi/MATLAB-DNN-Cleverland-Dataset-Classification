%hyperparameters
Ratio=0.4;  
layers=[24 20];
activations=["tanh","tanh"];
biasInit='zeros'; %Type of initial fully connected layer biases
weightInit='glorot' ;
lamda=1.8387e-06;
threshold=8; %the number of most important variables we keep in the model(max=13)

A = readmatrix('cleverland.csv');
x=size(A,1);
xx=linspace(1,x,x);

%pre-proccesing (removes rows with Nan values)
a=zeros(x,1);
for i=1:x
    a(i)=any(isnan(A(i,:)));
end
remove=find(a);
b=setdiff(xx,remove);
A=A(b,:);

x=size(A,1);%new size
xx=linspace(1,x,x);

%removes unimportant features from the data
Ax=A(:,1:end-1);
Ay=A(:,end);
idx = fscchi2(Ax,Ay);
d=zeros(13,1);
for i=1:13
    if(idx(i)>threshold)
       d(i)=1; 
    end
end
ln=linspace(1,13,13);
remove=find(d);
b=setdiff(ln,remove);
b=[b 14];
A=A(:,b);

Train = single.empty(0,8);
Test = single.empty(0,8);
Validate = single.empty(0,8);

%split into sets that have symmetrical ratios of each class
group=zeros(5,1);
for i = 1:5
    %oversampling
    k = find(A(:,end)==i-1); %indices of lines that belong to class i-1
    sizes=size(k,1);
    group=A(k,:); %the lines
    if (sizes<160) %size of most frequent class
       y = datasample(group,160);
       group=y;
    end
    
    sizes=160;
    xx=linspace(1,sizes,sizes);

    split=floor(sizes*Ratio);
    select = randperm(sizes,split); %choose random indices for validation&test data
    selectVal = select(1:floor(split/2));
    selectTest = select(floor(split/2):end);
    selectTrain =setdiff(xx,select); %the rest are training data
    group(selectVal,:);
    Validate=[Validate ;group(selectVal,:)];
    Test=[Test ; group(selectTest,:)];
    Train=[Train ;group(selectTrain,:)];

end

%histogram(Test(:,end))

%split predictor variables from class data
xVal=Validate(:,1:end-1);
yVal=Validate(:,end);
xTest=Test(:,1:end-1);
yTest=Test(:,end);
xTrain=Train(:,1:end-1);
yTrain=Train(:,end);

%create & train model
Mdl = fitcnet(xTrain,yTrain,"ValidationData",{xVal,yVal},"Verbose",1,'LayerSizes',layers,'Activations',activations,'LayerBiasesInitializer',biasInit ...
,'LayerWeightsInitializer',weightInit,'ValidationPatience',10,"Standardize",true,"Lambda",lamda);
%Mdl = fitcnet(xTrain,yTrain,"ValidationData",{xVal,yVal},'OptimizeHyperparameters','all');
testError = loss(Mdl,xTest,yTest, "LossFun","classiferror");
figure(1)
confusionchart(yTest,predict(Mdl,xTest))
figure(2)
iteration = Mdl.TrainingHistory.Iteration;
trainLosses = Mdl.TrainingHistory.TrainingLoss;
valLosses = Mdl.TrainingHistory.ValidationLoss;
plot(iteration,trainLosses,iteration,valLosses)
legend(["Training","Validation"])
xlabel("Iteration")
ylabel("Cross-Entropy Loss")
accuracy = 1 - testError;
fprintf('accuracy:%f',accuracy)


