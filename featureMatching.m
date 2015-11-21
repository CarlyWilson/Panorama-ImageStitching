function [ output_img ] = featureMatching(imgSet)
    img = rgb2gray(read(imgSet, 1));
    pts = detectSURFFeatures(img);
    [features, pts] = extractFeatures(img, pts);
    for i=2:imgSet.Count
        %Set prev image settings
        prevImg = img;
        prevPts = pts;
        prevFeatures = features;
        %read next image
        img = rgb2gray(read(imgSet, i));
        pts = detectSURFFeatures(img);
        [features, pts] = extractFeatures(img, pts);
        %Match points
        indexPairs = matchFeatures(features, prevFeatures, 'Unique', true);
        matchedPts1 =  pts(indexPairs(:,1), :);
        matchedPts2 = prevPts(indexPairs(:,2), :);
        figure(i); ax = axes; 
        showMatchedFeatures(img, prevImg, matchedPts1, matchedPts2,'montage', 'Parent', ax);  
    end
end