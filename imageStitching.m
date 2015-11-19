function [ output_img ] = imageStitching(imgSet)
%Pre: imgSet is a image set of pictures to be sitched together
%created with imageSet('IMAGE_DIR')
    montage(imgSet.ImageLocation);
    
    %Read the first image
    img = read(imgSet, 1);
    %convert to gray image
    grayImg = rgb2gray(img);
    %find intial features
    pts = detectSURFFeatures(grayImg);
    [features, pts] = extractFeatures(grayImg, pts);
    % Initialize all the transforms to the identity matrix using a project transform
    tforms(imgSet.Count) = projective2d(eye(3));
    
    %Go through the rest of the images
    for n = 2:imgSet.Count
        %Save the last points for comparison
        ptsPrevious = pts;
        featuresPrev = features;
        
        %Read next image and extract features of it
        img = read(imgSet, n);
        grayImg = rgb2gray(img);
        pts = detectSURFFeatures(grayImg);
        [features, pts] = extractFeatures(grayImg, pts);
        
        %Find matching pts between this image and the last one
        indexPairs = matchFeatures(features, featuresPrev, 'Unique', true);
        matchedPoints = pts(indexPairs(:,1), :);
        matchedPointsPrev = ptsPrevious(indexPairs(:,2), :);
        
        %Estimate the tranformation between this image and the last one
        %based on the the matching features
        tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
            'projective', 'Confidence', 99.9, 'MaxNumTrials', 5000);
        tforms(n).T =  tforms(n-1).T * tforms(n).T;
    end
    
    %Get the limits of each transform
    imageSize = size(img);
    for i = 1:numel(tforms)
        [xlim(i,:),ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    end
    %Use boundarys of limits to calculate which image should be in the center
    avgXLim = mean(xlim, 2);
    [~, idx] = sort(avgXLim);
    centerIdx = floor((numel(tforms)+1)/2);
    centerImageIdx = idx(centerIdx);
    %Apply the inverse tranform to all the other images
    Tinv = invert(tforms(centerImageIdx));
    for i = 1:numel(tforms)
        tforms(i).T = Tinv.T * tforms(i).T;
    end
    %Calculate the size of panorama based on the boundaries on the images
    for i = 1:numel(tforms)
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    end
    %find the min and max output limits for x and y
    xMin = min([1; xlim(:)]);
    xMax = max([imageSize(2); xlim(:)]);
    
    yMin = min([1; ylim(:)]);
    yMax = max([imageSize(1); ylim(:)]);
    %Create the panoramic based on the min and max boundaries
    width = round(xMax - xMin);
    height = round(yMax - yMin);
    panorama = zeros([height width 3], 'like', img);
    
    %Create blender to put images together
    blender = vision.AlphaBlender('Operation', 'Binary mask', ...
        'MaskSource', 'Input port');
    %Calculate the reverences between images
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    panoramaRef = imref2d([height width], xLimits, yLimits);
    %For every image in the set, put into panorama
    for i = 1:imgSet.Count
        img = read(imgSet, i);
        % transform images into the panorama
        warpedImage = imwarp(img, tforms(i), 'OutputView', panoramaRef);
        % Overlay the warpedImage onto the panorama
        panorama = step(blender, panorama, warpedImage, warpedImage(:,:,1));
    end
    
    output_img = panorama;
    figure
    imshow(output_img)
end

