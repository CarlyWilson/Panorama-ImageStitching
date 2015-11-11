function [ output_args ] = imageStitching(imgLoc)
    %Load Images
    imgDir = 'C:\Users\ZeLi\Desktop\CSC492\Panorama-ImageStitching\images\outside1';
    bDir =  fullfile(toolboxdir('vision'), 'visiondata', 'building');
    imgSet = imageSet(imgDir);
    montage(imgSet.ImageLocation);
    
    %Read the first image and Intialize features
    I = read(imgSet, 1);
    grayI = rgb2gray(I);
    points = detectSURFFeatures(grayI);
    [features, points] = extractFeatures(grayI, points);
    % Initialize all the transforms to the identity matrix using a project
    % transform
    tforms(imgSet.Count) = projective2d(eye(3));
    
    %Go through the rest of the images
    for n = 2:imgSet.Count
        pointsPrevious = points;
        featuresPrevious = features;
        
        %Read nth imagee and extract features
        I = read(imgSet, n);
        grayI = rgb2gray(I);
        points = detectSURFFeatures(grayI);
        [features, points] = extractFeatures(grayI, points);
        
        %Match points between n and n-1
        indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
        matchedPoints = points(indexPairs(:,1), :);
        matchedPointsPrev = pointsPrevious(indexPairs(:,2),:);
        
        %Estimate the tranformation between n and n-1
        tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
            'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
        tforms(n).T =  tforms(n-1).T * tforms(n).T;
    end
    
    %Get the limits of each transform
    imageSize = size(I);
    for i = 1:numel(tforms)
        [xlim(i,:),ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    end
    %Use limits to calculate center image
    avgXLim = mean(xlim, 2);
    [~, idx] = sort(avgXLim);
    centerIdx = floor((numel(tforms)+1)/2);
    centerImageIdx = idx(centerIdx);
    %Apply the inverse tranform to all the other images
    Tinv = invert(tforms(centerImageIdx));
    for i = 1.numel(tforms)
        tforms(i).T = Tinv.T * tforms(i).T;
    end
    %Calculate the size of panorama
    for i = 1:numel(tforms)
        [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(2)], [1 imageSize(1)]);
    end
    %find the min and max output limits
    xMin = min([1; xlim(:)]);
    xMax = max([imageSize(2); xlim(:)]);
    
    yMin = min([1; ylim(:)]);
    yMax = max([imageSize(1); ylim(:)]);
    %Create the panoramic
    width = round(xMax - xMin);
    height = round(yMax - yMin);
    panorama = zeros([height width 3], 'like', I);
    
    %Move the images into the panorama
    blender = vision.AlphaBlender('Operation', 'Binary mask', ...
        'MaskSource', 'Input port');
    
    xLimits = [xMin xMax];
    yLimits = [yMin yMax];
    panoramaView = imref2d([height width], xLimits, yLimits);
    
    for i = 1:imgSet.Count
        I = read(imgSet, i);
        % transform images into the pandorama
        warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
        % Overlay the warpedImage onto the panorama
        panorama = step(blender, panorama, warpedImage, warpedImage(:,:,1));
    end
    
    figure
    imshow(panorama)
end

