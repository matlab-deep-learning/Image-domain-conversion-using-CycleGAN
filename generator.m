function lgraph = generator(inputsize,outputsize,channel,EncoderDepth,numresnet)

% Generator function creates a layergraph deep learning model.
% This is used to translate a specific domain image to the other.
% lgraph = Generator(inputsize,outputsize,channel,EncoderDepth,numresnet) creates 
% lgraph including Encoder,Decoder and residual blocks.
% where
%     -inputsize: the size of input image (should be 128x128,256x256,512x512)
%     -outputsize: the size of output image
%     -channel: the number of channel of each layer.
%     -EncoderDepth: the number of convolutional layer of Encoder 
%     -numresnet: the number of residual block


% Copyright 2019-2020 The MathWorks, Inc.

% Encoder
    if sum([inputsize(1:2) outputsize(1:2)] == 512) == 4 |...
       sum([inputsize(1:2) outputsize(1:2)] == 256) == 4 |...
       sum([inputsize(1:2) outputsize(1:2)] == 128) == 4

        Depth = EncoderDepth;
    else
        error("Only 128x128xM,256x256xM and 512x512xM input/output size are supported"...
            +newline+ "And Both input and output should be the same size")
        return
    end


    layers1 = [
        imageInputLayer(inputsize,"Name","imageinput","Normalization","none")];
    for n=1:Depth
        tempconv =[
            convolution2dLayer([4 4],channel*2^(n-1),"Name","conv_"+num2str(n),"Stride",2,"Padding",[1 1 1 1])
            batchNormalizationLayer("Name","batchnorm_"+num2str(n))
            reluLayer("Name","relu_"+num2str(n))];
        layers1 = [
            layers1
            tempconv];
    end

    % Residual
    lgraph2 = layerGraph();
    for k=1:numresnet
        tempLayers = [
            convolution2dLayer([3 3],channel*2^(n-1),"Name","res_conv_"+num2str(k),"Padding",[1 1 1 1])
            batchNormalizationLayer("Name","res_batchnorm_"+num2str(k))
            reluLayer("Name","res_relu_"+num2str(k))];
        lgraph2 = addLayers(lgraph2,tempLayers);
        tempLayers = additionLayer(2,"Name","addition_"+num2str(k));
        lgraph2 = addLayers(lgraph2,tempLayers);

        if k==1
            lgraph2 = connectLayers(lgraph2,"res_relu_"+num2str(k),"addition_1/in1");
        else
            lgraph2 = connectLayers(lgraph2,"addition_"+num2str(k-1),"res_conv_"+num2str(k));
            lgraph2 = connectLayers(lgraph2,"addition_"+num2str(k-1),"addition_"+num2str(k)+"/in2");
            lgraph2 = connectLayers(lgraph2,"res_relu_"+num2str(k),"addition_"+num2str(k)+"/in1");
        end
    end

    % decorder
    layers3 = [];
    for m=1:Depth
        if m < Depth
            templayers3 = [
            transposedConv2dLayer([4 4],channel*2^(n-2),"Name","transposed-conv_"+num2str(m),"Stride",2,"Cropping","same")
            batchNormalizationLayer("Name","de_batchnorm_"+num2str(m))
            reluLayer("Name","de_relu_"+num2str(m))];
            layers3 = [
                layers3
                templayers3];
        elseif m == Depth
            templayers3 =[
            transposedConv2dLayer([4 4],outputsize(3),"Name","transposed-conv_"+num2str(m),"Stride",2,"Cropping","same")
            tanhLayer("Name","tanh")];
            layers3 = [
                layers3
                templayers3];        
        end
    end

    % integrate
    lgraph = lgraph2 ;
    lgraph = addLayers(lgraph,layers1);
    lgraph = addLayers(lgraph,layers3);

    lgraph = connectLayers(lgraph,"relu_"+num2str(n),"res_conv_1");
    lgraph = connectLayers(lgraph,"relu_"+num2str(m),"addition_1/in2");
    lgraph = connectLayers(lgraph,"addition_"+num2str(k),"transposed-conv_1");
    lgraph = dlnetwork(lgraph);
end

%% Copyright 2019 The MathWorks, Inc.