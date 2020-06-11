function lgraph = discriminator(inputSize,dDepth,conv_dim)
% Discriminator function creates a layergraph model.
% lgraph = Discriminator(inputSize,dDepth,conv_dim) creates a lgraph to classify input images.
% where
%     -inputSize: the size of input image
%     -dDepth: the number of convolution layer
%     -conv_dim: The size of channel for each layer
% The model created with this function is almost general structure of model to classify images.
% But it has array output instead of vector to calculate loss by each patch.(PatchGAN)
% 
% PatchGAN architecture was proposed in the following paper
% Reference
% Chuan Li and Michael Wand, "Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks".ECCV, 2016.
% https://arxiv.org/pdf/1604.04382.pdf

    % Copyright 2019-2020 The MathWorks, Inc.

    if (sum(inputSize(1:2) == 512) == 2 || ...
            sum(inputSize(1:2) == 256) == 2 || ...
            sum(inputSize(1:2) == 128) == 2) 
        Depth = dDepth;
    else
        error("Only 128x128xM,256x256xM and 512x512xM input/output size are supported"...
            +newline+ "And Both input and output should be the same size")
        return
    end

    if (inputSize(1)/2^Depth < 4) || Depth < 1 ||  Depth ~= floor(Depth)
        Dmax = log2(inputSize(1)/4);
        error("Depth should be 0 < Depth < "+num2str(Dmax)+" and an integer" )
    end

    layers = [imageInputLayer(inputSize,'Name','inputlayer',"Normalization","none")];
        for n=1:Depth
        if n==1
            templayers =[
            convolution2dLayer(4,conv_dim*2^(n-1),'Stride',2,"Name","conv_"+num2str(n),'Padding',1);
            reluLayer("Name","relu_"+num2str(n))];
            layers = [
                layers
                templayers];
        elseif n < Depth
            templayers =[
            convolution2dLayer(4,conv_dim*2^(n-1),'Stride',2,"Name","conv_"+num2str(n),'Padding',1);
            batchNormalizationLayer("Name","batch_"+num2str(n))
            reluLayer("Name","relu_"+num2str(n))];
            layers = [
                layers
                templayers];        
        elseif n==Depth
            templayers =[
            convolution2dLayer(1,1,'Stride',1,"Name","conv_"+num2str(n),'Padding',0,'WeightsInitializer', @(sz) 0.02*randn(sz, 'single'))];
            layers = [
                layers
                templayers];        
        end
        end
        lgraph = layerGraph(layers);
        lgraph = dlnetwork(lgraph);
end

