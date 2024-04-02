function SaveImgfcn(Mod_data, num_sample, num_images, Label, snr_value, p_offset_counter, Q_Gain_counter, img_p_offset_counter, T_delay,fc_offset_counter,datasetDir)
    img = figure('visible','off');
    set(img, 'Position', [0 0 224 224]);
    parfor i = 1:num_images
        dataSegment = zeros(1, num_sample);
        dataSegment = Mod_data((i-1)*num_sample+1:i*num_sample);
        scatterplot(dataSegment);
        %set(gcf, 'color','white');
        axis square; 
        axis off; 
        set(gca, 'position', [0 0 1 1], 'units', 'normalized');
        frame = getframe(gca);
        grayImage = rgb2gray(frame.cdata); 
        grayImage = imresize(grayImage, [224 224]);
        fileName = sprintf('%s_%02d%d%d%d%d%d%02d.png', Label,snr_value,p_offset_counter,Q_Gain_counter,img_p_offset_counter,T_delay,fc_offset_counter,i);
        fullPath = fullfile(datasetDir, fileName);
        imwrite(grayImage, fullPath); 
        cla;
    end
    % clear out Mod_data existingFiles existingIndices;
    close(img);
    % clear;
end